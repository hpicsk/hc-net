"""
Baseline ResNet for comparison with PCNN.

Standard ResNet implementation without Clifford algebra layers.
Used to measure the benefit of geometric feature interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Type


class BasicBlock(nn.Module):
    """Standard ResNet basic block."""
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
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


class Bottleneck(nn.Module):
    """ResNet bottleneck block."""
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64
    ):
        super().__init__()

        width = int(out_channels * (base_width / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class BaselineResNet(nn.Module):
    """
    Standard ResNet for CIFAR-100.

    Modified for 32x32 images (no initial maxpool, smaller stride).
    """

    def __init__(
        self,
        block: Type[nn.Module],
        layers: List[int],
        num_classes: int = 100,
        base_width: int = 64,
        zero_init_residual: bool = False
    ):
        super().__init__()

        self.inplanes = base_width

        # Modified stem for CIFAR (no 7x7 conv, no maxpool)
        self.conv1 = nn.Conv2d(3, base_width, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)

        # Residual stages
        self.layer1 = self._make_layer(block, base_width, layers[0])
        self.layer2 = self._make_layer(block, base_width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_width * 8, layers[3], stride=2)

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 8 * block.expansion, num_classes)

        self._initialize_weights(zero_init_residual)

    def _make_layer(
        self,
        block: Type[nn.Module],
        planes: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self, zero_init_residual: bool):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def get_features(self, x: torch.Tensor) -> dict:
        """Extract intermediate features for analysis."""
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


class BaselineResNetSmall(nn.Module):
    """
    Smaller baseline ResNet matching PCNNResNetSmall architecture.
    """

    def __init__(self, num_classes: int = 100, base_width: int = 64):
        super().__init__()

        # Stem
        self.conv1 = nn.Conv2d(3, base_width, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)

        # Stage 1: 32x32, 64 channels
        self.stage1 = nn.Sequential(
            BasicBlock(base_width, base_width),
            BasicBlock(base_width, base_width),
        )

        # Stage 2: 16x16, 128 channels
        self.stage2 = nn.Sequential(
            BasicBlock(base_width, base_width * 2, stride=2,
                      downsample=nn.Sequential(
                          nn.Conv2d(base_width, base_width * 2, 1, stride=2, bias=False),
                          nn.BatchNorm2d(base_width * 2)
                      )),
            BasicBlock(base_width * 2, base_width * 2),
        )

        # Stage 3: 8x8, 256 channels
        self.stage3 = nn.Sequential(
            BasicBlock(base_width * 2, base_width * 4, stride=2,
                      downsample=nn.Sequential(
                          nn.Conv2d(base_width * 2, base_width * 4, 1, stride=2, bias=False),
                          nn.BatchNorm2d(base_width * 4)
                      )),
            BasicBlock(base_width * 4, base_width * 4),
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


def baseline_resnet18(num_classes: int = 100) -> BaselineResNet:
    """Baseline ResNet-18."""
    return BaselineResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def baseline_resnet34(num_classes: int = 100) -> BaselineResNet:
    """Baseline ResNet-34."""
    return BaselineResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def baseline_resnet50(num_classes: int = 100) -> BaselineResNet:
    """Baseline ResNet-50."""
    return BaselineResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
