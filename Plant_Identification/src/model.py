"""
FloraNet-V3: A Bespoke Deep Convolutional Neural Network for Plant Identification.
Built from the ground up using the MobileNetV3 architectural blueprint:
  - Inverted Residual Bottlenecks with Linear Manifolds
  - Depthwise Separable Convolutions
  - Squeeze-and-Excitation (SE) Channel-Wise Attention
  - Hard-Swish and Hard-Sigmoid Non-Linearities
  - Kaiming Normal (He) Weight Initialization
"""

import math
import sys
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config

# -------------------------------------------------------------------------
# 1. Custom Activation Functions (MobileNetV3 Blueprint)
# -------------------------------------------------------------------------

class HardSigmoid(nn.Module):
    """
    Piecewise linear approximation of sigmoid:
    h-sigmoid(x) = ReLU6(x + 3) / 6
    """
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + 3.0) / 6.0


class HardSwish(nn.Module):
    """
    Hardware-efficient approximation of Swish activation:
    h-swish(x) = x * (ReLU6(x + 3) / 6)
    """
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.relu(x + 3.0) / 6.0)


# -------------------------------------------------------------------------
# 2. Squeeze-and-Excitation (SE) Attention Module
# -------------------------------------------------------------------------

class SqueezeExcitation(nn.Module):
    """
    Channel-wise attention module:
    Pools global spatial context, performs channel dimensionality reduction,
    applies Hard-Sigmoid gating, and dynamically recalibrates channel activations.
    """
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=True),
            HardSigmoid(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x)
        return x * scale


# -------------------------------------------------------------------------
# 3. Inverted Residual Bottleneck Block
# -------------------------------------------------------------------------

def _make_divisible(v: int, divisor: int = 8, min_value: int = None) -> int:
    """Ensures all channel counts are divisible by divisor for hardware alignment."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    """
    Inverted Residual Block:
      1. 1x1 Pointwise Conv (expands channel dimension)
      2. KxK Depthwise Separable Conv (spatial feature extraction)
      3. Squeeze-and-Excitation (SE attention recalibration)
      4. 1x1 Pointwise Conv (linear projection back to low-dim manifold)
      5. Residual shortcut (if stride == 1 and in_c == out_c)
    """
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
        # 1. Expansion (Pointwise)
        if exp_channels != in_channels:
            layers.extend([
                nn.Conv2d(in_channels, exp_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(exp_channels),
                activation(inplace=True)
            ])

        # 2. Depthwise Convolution
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

        # 3. Squeeze-and-Excitation Attention
        if use_se:
            layers.append(SqueezeExcitation(exp_channels))

        # 4. Projection (Pointwise Linear Bottleneck)
        layers.extend([
            nn.Conv2d(exp_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
            # Notice: No non-linearity here, preserving linear feature manifold!
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


# -------------------------------------------------------------------------
# 4. FloraNet-V3 Architecture (Bespoke Model)
# -------------------------------------------------------------------------

class FloraNetV3(nn.Module):
    """
    FloraNet-V3: Bespoke Deep Neural Network for Plant Species Identification.
    Architectural blueprint derived from MobileNetV3, written and trained from scratch.
    """
    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        dropout_rate: float = config.DROPOUT_RATE,
        width_mult: float = 1.0
    ):
        super().__init__()
        self.num_classes = num_classes

        # Specification: [kernel_size, exp_size, out_channels, use_se, use_hs, stride]
        cfgs = [
            # Stage 1
            [3, 16, 16, False, False, 1],
            # Stage 2
            [3, 64, 24, False, False, 2],
            [3, 72, 24, False, False, 1],
            # Stage 3
            [5, 72, 40, True, False, 2],
            [5, 120, 40, True, False, 1],
            [5, 120, 40, True, False, 1],
            # Stage 4
            [3, 240, 80, False, True, 2],
            [3, 200, 80, False, True, 1],
            [3, 184, 80, False, True, 1],
            [3, 184, 80, False, True, 1],
            # Stage 5
            [3, 480, 112, True, True, 1],
            [3, 672, 112, True, True, 1],
            # Stage 6
            [5, 672, 160, True, True, 2],
            [5, 960, 160, True, True, 1],
            [5, 960, 160, True, True, 1],
        ]

        # 1. Stem Layer: Conv 3x3 with stride 2
        input_channel = _make_divisible(16 * width_mult)
        features = [
            nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            HardSwish(inplace=True)
        ]

        # 2. Inverted Residual Bottleneck Stages
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

        # 3. Final Feature Aggregation Conv (1x1)
        last_conv_channel = _make_divisible(960 * width_mult)
        features.extend([
            nn.Conv2d(input_channel, last_conv_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_conv_channel),
            HardSwish(inplace=True)
        ])
        self.features = nn.Sequential(*features)

        # 4. Dense Classification Head
        last_channel = _make_divisible(1280 * width_mult)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(last_conv_channel, last_channel),
            HardSwish(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(last_channel, num_classes)
        )

        # 5. Initialize all weights with Kaiming Normal
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """
        Kaiming Normal (He) initialization:
        Optimal initialization for ReLU and Swish networks trained from scratch.
        """
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


# -------------------------------------------------------------------------
# Helper Construction Function
# -------------------------------------------------------------------------

def build_model(
    num_classes: int = config.NUM_CLASSES,
    dropout_rate: float = config.DROPOUT_RATE,
    pretrained: bool = False
) -> nn.Module:
    """
    Constructs the bespoke FloraNet-V3 network.
    """
    model = FloraNetV3(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        width_mult=1.0
    )
    return model
