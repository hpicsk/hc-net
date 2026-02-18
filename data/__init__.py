"""
Data loading utilities for HC-Net.

Provides datasets for:
- N-body physics simulations (2D and 3D)
- Spinning N-body systems (2D and 3D)
- Chiral spiral point clouds
- MD17 molecular dynamics
- CIFAR-100 with compositional splits
- Geometric MNIST (2-way binding)
- Relational MNIST (3-way binding)
"""

from .nbody_dataset import NBodyDataset, get_nbody_loaders_with_ood
from .spinning_nbody import SpinningNBodyDataset, get_spinning_nbody_loaders
from .spinning_nbody_3d import SpinningChiralityDataset3D
from .chiral_spirals import ChiralSpiralDataset
from .md17_dataset import MD17Dataset, MD17_MOLECULES, get_md17_loaders
from .cifar100 import CIFAR100Dataset, get_cifar100_loaders
from .compositional import create_compositional_split, CompositionallySplitCIFAR100
from .geometric_mnist import GeometricMNISTDataset, get_geometric_mnist_loaders
from .relational_mnist import RelationalMNISTDataset, get_relational_mnist_loaders

__all__ = [
    'NBodyDataset',
    'get_nbody_loaders_with_ood',
    'SpinningNBodyDataset',
    'get_spinning_nbody_loaders',
    'SpinningChiralityDataset3D',
    'ChiralSpiralDataset',
    'MD17Dataset',
    'MD17_MOLECULES',
    'get_md17_loaders',
    'CIFAR100Dataset',
    'get_cifar100_loaders',
    'create_compositional_split',
    'CompositionallySplitCIFAR100',
    'GeometricMNISTDataset',
    'get_geometric_mnist_loaders',
    'RelationalMNISTDataset',
    'get_relational_mnist_loaders',
]
