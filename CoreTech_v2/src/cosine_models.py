"""
Shared Foundation Model Architectures for CoreTech v2:
  1. CosineNormalizedHead: L2-normalized embeddings & prototypes with bias=0 to eliminate logit hijacking
     and provide reliable Cosine Similarity scores for Open-Set Unknown/Genus gating.
  2. BotanicalBackboneWrapper: Supports BioCLIP-2 (open_clip), EVA-02 / DINOv2 (timm), and ConvNeXt-Base (timm).
  3. MultiTileAttentionMIL: Global + multi-tile surface pathology model with auxiliary
     [Healthy | Biotic | Abiotic] super-head.
"""
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineNormalizedHead(nn.Module):
    """L2-normalized prototype classifier head (bias-free, temperature-scaled cosine logits)."""
    def __init__(self, in_features: int, num_classes: int, init_scale: float = 20.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features) * 0.02)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(init_scale)))

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_norm = F.normalize(features, p=2, dim=-1)
        w_norm = F.normalize(self.weight, p=2, dim=-1)
        cos_sim = F.linear(feat_norm, w_norm)
        scale = torch.clamp(self.logit_scale.exp(), min=5.0, max=45.0)
        return cos_sim * scale, cos_sim


class BioCLIP2Classifier(nn.Module):
    """BioCLIP-2 ViT-L/14 botanical classifier with CosineNormalizedHead."""
    def __init__(self, num_classes: int, unfreeze_last_blocks: int = 12, pretrained: bool = True):
        super().__init__()
        import open_clip
        pretrained_tag = "hf-hub:imageomics/bioclip-2"
        clip_model, _, _ = open_clip.create_model_and_transforms(pretrained_tag)
        self.visual = clip_model.visual
        # Infer visual output dim
        embed_dim = getattr(self.visual, "output_dim", 768)
        self.drop = nn.Dropout(0.25)
        self.head = CosineNormalizedHead(embed_dim, num_classes, init_scale=22.0)

        if not pretrained:
            pass
        # Freeze early blocks to preserve TreeOfLife-200M botanical prior, fine-tune last N blocks cleanly
        for p in self.visual.parameters():
            p.requires_grad = False
        if hasattr(self.visual, "transformer") and hasattr(self.visual.transformer, "resblocks"):
            blocks = self.visual.transformer.resblocks
            for blk in blocks[-unfreeze_last_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
        for attr in ["ln_post", "proj"]:
            if hasattr(self.visual, attr):
                obj = getattr(self.visual, attr)
                if isinstance(obj, nn.Parameter):
                    obj.requires_grad = True
                elif isinstance(obj, nn.Module):
                    for p in obj.parameters():
                        p.requires_grad = True

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.drop(self.forward_features(x))
        return self.head(feats)


class TimmCosineClassifier(nn.Module):
    """Timm vision backbone (EVA-02, DINOv2, ConvNeXt-Base) with CosineNormalizedHead."""
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.25):
        super().__init__()
        import timm
        self.model_name = model_name
        extra_kw = {"dynamic_img_size": True} if "dinov2" in model_name.lower() else {}
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, **extra_kw)
        in_features = self.backbone.num_features
        self.drop = nn.Dropout(drop_rate)
        self.head = CosineNormalizedHead(in_features, num_classes, init_scale=20.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.drop(self.forward_features(x))
        return self.head(feats)


class MultiTileAttentionMIL(nn.Module):
    """
    Dual-Scale Surface Pathology Network:
    Takes bag shape [B, N_tiles, C, H, W] (global view + native-res surface tiles),
    aggregates tile embeddings via gated attention, and outputs:
      - fine_logits [B, 19] (cosine-normalized surface class)
      - super_logits [B, 3] (Healthy vs Biotic Pathology vs Abiotic Stress)
      - tile_weights [B, N_tiles] (explainable attention weights highlighting which tiles show disease)
    """
    def __init__(self, model_name: str, num_classes: int = 19, pretrained: bool = True):
        super().__init__()
        import timm
        extra_kw = {"dynamic_img_size": True} if "dinov2" in model_name.lower() else {}
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, **extra_kw)
        dim = self.backbone.num_features

        self.attn_V = nn.Linear(dim, 256)
        self.attn_U = nn.Linear(dim, 256)
        self.attn_w = nn.Linear(256, 1)

        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        self.fine_head = CosineNormalizedHead(dim, num_classes, init_scale=20.0)
        self.super_head = nn.Linear(dim, 3)  # 0=Healthy, 1=Biotic, 2=Abiotic

    def forward(self, bag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n_tiles, c, h, w = bag.shape
        flat = bag.view(b * n_tiles, c, h, w)
        feats = self.backbone(flat).view(b, n_tiles, -1)  # [B, N, D]

        global_feat = feats[:, 0, :]  # View 0 = global plant context
        # Attention pooling across all views (including surface tiles)
        a_v = torch.tanh(self.attn_V(feats))
        a_u = torch.sigmoid(self.attn_U(feats))
        attn_logits = self.attn_w(a_v * a_u).squeeze(-1)  # [B, N]
        attn_weights = F.softmax(attn_logits, dim=-1)
        pooled_feat = torch.bmm(attn_weights.unsqueeze(1), feats).squeeze(1)

        fused = self.fusion(torch.cat([global_feat, pooled_feat], dim=-1))
        fine_logits, cos_sim = self.fine_head(fused)
        super_logits = self.super_head(fused)
        return fine_logits, super_logits, cos_sim, attn_weights
