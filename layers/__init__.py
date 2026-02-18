"""
HC-Net Neural Network Layers.

Provides Clifford algebra-aware neural network layers:
- CliffordLinear: Linear layer with geometric product
- CliffordBatchNorm: Batch normalization for multivectors
- CliffordReLU: Grade-aware activation function
- EGNN layers for E(n) equivariant baselines
- LocalMPNN for k-NN message passing
"""

from .linear import CliffordLinear
from .norm import CliffordBatchNorm, CliffordLayerNorm
from .activation import CliffordReLU, CliffordGELU, MVSiLU
from .egnn_layers import EGNNLayer, EGNNBlock, VelocityEGNNLayer
from .local_mpnn import LocalMPNNLayer

__all__ = [
    'CliffordLinear',
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
