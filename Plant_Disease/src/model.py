"""
FloraNet-Hierarchical: Multi-Head MobileNetV3 Architecture for Grouped Plant Pathology.
Consists of:
  - Shared Botanical Feature Backbone (MobileNetV3 Inverted Residuals + SE Attention)
  - Stage 1: Plant Species Routing Head (5 classes: Aloe, Cactus, Money Plant, Snake Plant, Spider Plant)
  - Stage 2: Specialized Pathology Heads per Plant Species Group
      * Aloe Head: 5 conditions (Healthy vs. Anthracnose, LeafSpot, Rust, Sunburn)
      * Cactus Head: 2 conditions (Healthy vs. Cochineal Scale Infestation)
      * Money Plant Head: 3 conditions (Healthy vs. Bacterial Wilt, Manganese Toxicity)
      * Snake Plant Head: 3 conditions (Healthy vs. Anthracnose, Leaf Withering)
      * Spider Plant Head: 3 conditions (Healthy vs. Fungal Leaf Spot, Leaf Tip Necrosis)
"""

import math
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from Plant_Disease.src import config
except (ImportError, ModuleNotFoundError):
    from src import config

# -------------------------------------------------------------------------
# 1. Custom Activation Functions (MobileNetV3 Blueprint)
# -------------------------------------------------------------------------

class HardSigmoid(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + 3.0) / 6.0

class HardSwish(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.relu(x + 3.0) / 6.0)

# -------------------------------------------------------------------------
# 2. Squeeze-and-Excitation (SE) Channel-Wise Attention
# -------------------------------------------------------------------------

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(1, in_channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, in_channels, kernel_size=1, bias=True),
            HardSigmoid(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)

# -------------------------------------------------------------------------
# 3. Inverted Residual Bottleneck Block
# -------------------------------------------------------------------------

def _make_divisible(v: int, divisor: int = 8, min_value: int = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        exp_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
        use_hs: bool
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = (self.stride == 1 and in_channels == out_channels)
        activation = HardSwish if use_hs else nn.ReLU

        layers = []
        if exp_channels != in_channels:
            layers.extend([
                nn.Conv2d(in_channels, exp_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(exp_channels),
                activation(inplace=True)
            ])

        padding = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv2d(
                exp_channels,
                exp_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=exp_channels,
                bias=False
            ),
            nn.BatchNorm2d(exp_channels),
            activation(inplace=True)
        ])

        if use_se:
            layers.append(SqueezeExcitation(exp_channels))

        layers.extend([
            nn.Conv2d(exp_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)

# -------------------------------------------------------------------------
# 4. FloraNet-Hierarchical Architecture
# -------------------------------------------------------------------------

class FloraNetHierarchical(nn.Module):
    """
    Multi-Head MobileNetV3 for Hierarchical Plant Disease Diagnosis.
    Features:
      - Shared botanical feature backbone.
      - Species Router: Identifies which of the 5 plant groups the leaf belongs to.
      - 5 Specialized Disease Heads: Eliminates species confusion by evaluating pathology
        strictly within the relevant plant group.
    """
    def __init__(
        self,
        plant_groups: dict = config.PLANT_GROUPS,
        dropout_rate: float = config.DROPOUT_RATE,
        width_mult: float = 1.0
    ):
        super().__init__()
        self.plant_groups = plant_groups
        self.species_names = list(plant_groups.keys())
        self.num_species = len(self.species_names)

        # MobileNetV3 specification: [kernel_size, exp_size, out_channels, use_se, use_hs, stride]
        cfgs = [
            [3, 16, 16, False, False, 1],
            [3, 64, 24, False, False, 2],
            [3, 72, 24, False, False, 1],
            [5, 72, 40, True, False, 2],
            [5, 120, 40, True, False, 1],
            [5, 120, 40, True, False, 1],
            [3, 240, 80, False, True, 2],
            [3, 200, 80, False, True, 1],
            [3, 184, 80, False, True, 1],
            [3, 184, 80, False, True, 1],
            [3, 480, 112, True, True, 1],
            [3, 672, 112, True, True, 1],
            [5, 672, 160, True, True, 2],
            [5, 960, 160, True, True, 1],
            [5, 960, 160, True, True, 1],
        ]

        # 1. Shared Backbone: Stem + Inverted Residuals
        input_channel = _make_divisible(16 * width_mult)
        features = [
            nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            HardSwish(inplace=True)
        ]

        for k, exp_size, c, se, hs, s in cfgs:
            output_channel = _make_divisible(c * width_mult)
            exp_channel = _make_divisible(exp_size * width_mult)
            features.append(
                InvertedResidual(
                    in_channels=input_channel,
                    exp_channels=exp_channel,
                    out_channels=output_channel,
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    use_hs=hs
                )
            )
            input_channel = output_channel

        # Final Feature Aggregation Conv (1x1)
        last_conv_channel = _make_divisible(960 * width_mult)
        features.extend([
            nn.Conv2d(input_channel, last_conv_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_conv_channel),
            HardSwish(inplace=True)
        ])
        self.features = nn.Sequential(*features)

        # 2. Shared Dense Embedding Projection
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        last_channel = _make_divisible(1280 * width_mult)
        self.embedding = nn.Sequential(
            nn.Linear(last_conv_channel, last_channel),
            HardSwish(inplace=True),
            nn.Dropout(p=dropout_rate)
        )

        # 3. Stage 1: Plant Species Routing Head (5 classes)
        self.species_head = nn.Linear(last_channel, self.num_species)

        # 4. Stage 2: Specialized Pathology Heads for Each Plant Group
        self.disease_heads = nn.ModuleDict({
            sp_name: nn.Linear(last_channel, len(conds))
            for sp_name, conds in plant_groups.items()
        })

        self._initialize_weights()

    def extract_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns 2D spatial feature maps [B, 960, 7, 7] for Grad-CAM spot localizer."""
        return self.features(x)

    def forward(self, x: torch.Tensor, target_species: str = None):
        """
        Forward pass.
        Returns:
            species_logits: [B, 5] (Plant type probabilities)
            disease_outputs: dict of {species_name: disease_logits}
        """
        feats = self.features(x)
        pooled = self.pool(feats)
        flat = self.flatten(pooled)
        emb = self.embedding(flat)

        species_logits = self.species_head(emb)

        disease_outputs = {}
        if target_species and target_species in self.disease_heads:
            disease_outputs[target_species] = self.disease_heads[target_species](emb)
        else:
            for sp_name, head in self.disease_heads.items():
                disease_outputs[sp_name] = head(emb)

        return species_logits, disease_outputs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

def build_model(
    plant_groups: dict = config.PLANT_GROUPS,
    dropout_rate: float = config.DROPOUT_RATE,
    pretrained_backbone_path: Path = None
) -> FloraNetHierarchical:
    """Constructs the FloraNet-Hierarchical model with optional botanical backbone pretraining."""
    model = FloraNetHierarchical(
        plant_groups=plant_groups,
        dropout_rate=dropout_rate,
        width_mult=1.0
    )

    if pretrained_backbone_path and Path(pretrained_backbone_path).exists():
        print(f"[*] Loading pretrained botanical backbone from: {pretrained_backbone_path}")
        checkpoint = torch.load(pretrained_backbone_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        features_dict = {
            k: v for k, v in state_dict.items()
            if k.startswith("features.") and k in model.state_dict()
        }
        model.load_state_dict(features_dict, strict=False)
        print(f"  Loaded {len(features_dict)} botanical layers into FloraNet-Hierarchical backbone.")

    return model
