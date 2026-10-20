"""
PhytoCare Model Architecture Module
Supports ConvNeXt-Small and Swin-Transformer-Tiny backbones.
Enables transferring botanical feature weights directly from FloraNet-250.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src import config

def build_disease_model(
    num_classes: int = config.NUM_CLASSES,
    backbone: str = "convnext_small",
    transfer_floranet_weights: bool = True,
    dropout_rate: float = 0.25
) -> nn.Module:
    """
    Builds the pathology model architecture.
    Optionally initializes the backbone with FloraNet-250 botanical weights.
    """
    if backbone == "convnext_small":
        model = models.convnext_small(weights=None)
        
        # 1. Attempt transfer learning from FloraNet-250 botanical weights
        if transfer_floranet_weights and config.FLORANET_IDENTIFICATION_WEIGHTS.exists():
            try:
                ckpt = torch.load(config.FLORANET_IDENTIFICATION_WEIGHTS, map_location="cpu", weights_only=False)
                state = ckpt.get("model_state_dict", {})
                # Filter out old 250-class classifier head
                filtered_state = {k: v for k, v in state.items() if not k.startswith("classifier.2")}
                model.load_state_dict(filtered_state, strict=False)
                print(f"[Model Transfer] Successfully loaded botanical features from FloraNet-250 ({config.FLORANET_IDENTIFICATION_WEIGHTS})")
            except Exception as e:
                print(f"[Model Transfer Warning] Could not load FloraNet-250 weights: {e}, falling back to ImageNet.")
                model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        else:
            model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)

        # Replace classification head with 16-class pathology classifier
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        return model

    elif backbone == "swin_t":
        model = models.swin_t(weights=None)
        
        if transfer_floranet_weights and config.FLORANET_SWIN_WEIGHTS.exists():
            try:
                ckpt = torch.load(config.FLORANET_SWIN_WEIGHTS, map_location="cpu", weights_only=False)
                state = ckpt.get("model_state_dict", {})
                filtered_state = {k: v for k, v in state.items() if not k.startswith("head")}
                model.load_state_dict(filtered_state, strict=False)
                print(f"[Model Transfer] Successfully loaded botanical features from Swin-384 FloraNet ({config.FLORANET_SWIN_WEIGHTS})")
            except Exception as e:
                print(f"[Model Transfer Warning] Could not load Swin FloraNet weights: {e}, falling back to ImageNet.")
                model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        else:
            model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)

        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        return model

    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

