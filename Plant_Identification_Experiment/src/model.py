"""
Model architecture module for Plant Identification Experimentation.
Supports both Transfer Learning (ImageNet-1k pretrained MobileNetV3 Large)
and From-Scratch bespoke FloraNet-V3 for direct experimental comparison.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config

def build_model(
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = config.USE_PRETRAINED,
    dropout_rate: float = config.DROPOUT_RATE,
    backbone: str = "mobilenet_v3_large"
) -> nn.Module:
    """
    Builds the model architecture.
    Supports:
      - 'mobilenet_v3_large' (Efficient mobile baseline)
      - 'convnext_small' (Flagship 50M parameter vision backbone)
      - 'convnext_tiny' (28M parameter vision backbone)
    """
    if backbone == "convnext_small":
        weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        model = models.convnext_small(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        return model
    elif backbone == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        return model
    elif backbone == "swin_t":
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        model = models.swin_t(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        return model
    elif backbone == "swin_s":
        weights = models.Swin_S_Weights.DEFAULT if pretrained else None
        model = models.swin_s(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        return model
    else:
        if pretrained:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
            model = models.mobilenet_v3_large(weights=weights)
        else:
            model = models.mobilenet_v3_large(weights=None)
            
        in_features = model.classifier[3].in_features
        model.classifier[2] = nn.Dropout(p=dropout_rate, inplace=True)
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model

def set_parameter_requires_grad(model: nn.Module, freeze_backbone: bool = True):
    """Freezes or unfreezes backbone feature extraction layers for staged fine-tuning."""
    for param in model.features.parameters():
        param.requires_grad = not freeze_backbone

def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

