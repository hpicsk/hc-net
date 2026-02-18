"""
HC-Net ResNet: ResNet with Clifford Algebra Layers (Hybrid Clifford Network).

This model uses BlockCliffordConv2d layers to capture
geometric feature interactions as bivector components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Type

from ..layers.conv import BlockCliffordConv2d, HybridCliffordConv2d
from ..layers.norm import CliffordBatchNorm
from ..layers.activation import CliffordReLU
from ..layers.linear import ProjectionHead


class CliffordBasicBlock(nn.Module):
    """
    Basic residual block with Clifford convolutions.

    Structure: x -> Conv -> BN -> ReLU -> Conv -> BN -> + x -> ReLU
    """
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        block_size: int = 8,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()

        self.conv1 = BlockCliffordConv2d(
            in_channels, out_channels,
            kernel_size=3, block_size=block_size,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = BlockCliffordConv2d(
            out_channels, out_channels,
            kernel_size=3, block_size=block_size,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class HybridBasicBlock(nn.Module):
    """
    Residual block with hybrid Clifford convolutions.

    Uses HybridCliffordConv2d which is more efficient while
    still capturing geometric interactions.
    """
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        block_size: int = 8,
        downsample: Optional[nn.Module] = None,
        interaction_ratio: float = 0.1
    ):
        super().__init__()

        self.conv1 = HybridCliffordConv2d(
            in_channels, out_channels,
            kernel_size=3, block_size=block_size,
            stride=stride, padding=1, bias=False,
            interaction_ratio=interaction_ratio
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = HybridCliffordConv2d(
            out_channels, out_channels,
            kernel_size=3, block_size=block_size,
            stride=1, padding=1, bias=False,
            interaction_ratio=interaction_ratio
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class HCNetResNet(nn.Module):
    """
    ResNet with Clifford Algebra layers (Hybrid Clifford Network).

    Architecture follows standard ResNet but replaces Conv2d
    with BlockCliffordConv2d in residual blocks.

    Args:
        block: Block type (CliffordBasicBlock or HybridBasicBlock)
        layers: Number of blocks per stage [stage1, stage2, stage3, stage4]
        num_classes: Number of output classes
        block_size: Clifford algebra dimension
        base_width: Base channel width
    """

    def __init__(
        self,
        block: Type[nn.Module],
        layers: List[int],
        num_classes: int = 100,
        block_size: int = 8,
        base_width: int = 64,
        zero_init_residual: bool = False
    ):
        super().__init__()

        self.block_size = block_size
        self.inplanes = base_width

        # Initial convolution (standard, not Clifford)
        self.conv1 = nn.Conv2d(3, base_width, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)

        # Residual stages
        self.layer1 = self._make_layer(block, base_width, layers[0],
                                       block_size=block_size)
        self.layer2 = self._make_layer(block, base_width * 2, layers[1],
                                       stride=2, block_size=block_size)
        self.layer3 = self._make_layer(block, base_width * 4, layers[2],
                                       stride=2, block_size=block_size)
        self.layer4 = self._make_layer(block, base_width * 8, layers[3],
                                       stride=2, block_size=block_size)

        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 8 * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights(zero_init_residual)

    def _make_layer(
        self,
        block: Type[nn.Module],
        planes: int,
        blocks: int,
        stride: int = 1,
        block_size: int = 8
    ) -> nn.Sequential:
        """Create a residual stage."""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride,
                           block_size=block_size, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, block_size=block_size))

        return nn.Sequential(*layers)

    def _initialize_weights(self, zero_init_residual: bool):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, (CliffordBasicBlock, HybridBasicBlock)):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def get_bivector_features(self, x: torch.Tensor) -> dict:
        """
        Extract bivector features from each layer for analysis.

        Returns intermediate activations for interpretability.
        """
        features = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        features['layer1'] = x.clone()

        x = self.layer2(x)
        features['layer2'] = x.clone()

        x = self.layer3(x)
        features['layer3'] = x.clone()

        x = self.layer4(x)
        features['layer4'] = x.clone()

        return features


class HCNetResNetSmall(nn.Module):
    """
    Smaller HC-Net ResNet for faster experiments.

    Suitable for CIFAR-100 with 32x32 images.
    """

    def __init__(
        self,
        num_classes: int = 100,
        block_size: int = 8,
        base_width: int = 64
    ):
        super().__init__()

        self.block_size = block_size

        # Stem
        self.conv1 = nn.Conv2d(3, base_width, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)

        # Stage 1: 32x32, 64 channels
        self.stage1 = nn.Sequential(
            HybridBasicBlock(base_width, base_width, block_size=block_size),
            HybridBasicBlock(base_width, base_width, block_size=block_size),
        )

        # Stage 2: 16x16, 128 channels
        self.stage2 = nn.Sequential(
            HybridBasicBlock(base_width, base_width * 2, stride=2,
                            block_size=block_size,
                            downsample=nn.Sequential(
                                nn.Conv2d(base_width, base_width * 2, 1, stride=2, bias=False),
                                nn.BatchNorm2d(base_width * 2)
                            )),
            HybridBasicBlock(base_width * 2, base_width * 2, block_size=block_size),
        )

        # Stage 3: 8x8, 256 channels
        self.stage3 = nn.Sequential(
            HybridBasicBlock(base_width * 2, base_width * 4, stride=2,
                            block_size=block_size,
                            downsample=nn.Sequential(
                                nn.Conv2d(base_width * 2, base_width * 4, 1, stride=2, bias=False),
                                nn.BatchNorm2d(base_width * 4)
                            )),
            HybridBasicBlock(base_width * 4, base_width * 4, block_size=block_size),
        )

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 4, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# Backward compatibility aliases
PCNNResNet = HCNetResNet
PCNNResNetSmall = HCNetResNetSmall


def hcnet_resnet18(num_classes: int = 100, block_size: int = 8) -> HCNetResNet:
    """HC-Net-ResNet-18 configuration."""
    return HCNetResNet(HybridBasicBlock, [2, 2, 2, 2],
                      num_classes=num_classes, block_size=block_size)


def hcnet_resnet34(num_classes: int = 100, block_size: int = 8) -> HCNetResNet:
    """HC-Net-ResNet-34 configuration."""
    return HCNetResNet(HybridBasicBlock, [3, 4, 6, 3],
                      num_classes=num_classes, block_size=block_size)


# Backward compatibility
pcnn_resnet18 = hcnet_resnet18
pcnn_resnet34 = hcnet_resnet34
