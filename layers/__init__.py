"""
HC-Net Neural Network Layers.

Provides Clifford algebra-aware neural network layers:
- CliffordLinear: Linear layer with geometric product
- BlockCliffordConv2d: Convolutional layer with block-wise Clifford operations
- CliffordBatchNorm: Batch normalization for multivectors
- CliffordReLU: Grade-aware activation function
- EGNN layers for E(n) equivariant baselines
- LocalMPNN for k-NN message passing
"""

from .linear import CliffordLinear
from .conv import BlockCliffordConv2d
from .norm import CliffordBatchNorm, CliffordLayerNorm
from .activation import CliffordReLU, CliffordGELU, MVSiLU
from .egnn_layers import EGNNLayer, EGNNBlock, VelocityEGNNLayer
from .local_mpnn import LocalMPNNLayer

__all__ = [
    'CliffordLinear',
    'BlockCliffordConv2d',
    'CliffordBatchNorm',
    'CliffordLayerNorm',
    'CliffordReLU',
    'CliffordGELU',
    'MVSiLU',
    'EGNNLayer',
    'EGNNBlock',
    'VelocityEGNNLayer',
    'LocalMPNNLayer',
]
