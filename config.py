"""
Configuration for HC-Net experiments.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrainingConfig:
    """Training configuration for physics/molecular experiments."""
    # Model
    model_type: str = 'hcnet'
    hidden_dim: int = 128
    n_layers: int = 4
    dropout: float = 0.1

    # Data
    n_particles: int = 5
    batch_size: int = 128
    num_workers: int = 4

    # Training
    epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 1e-5
    optimizer: str = 'adamw'

    # Learning rate schedule
    scheduler: str = 'cosine'
    warmup_epochs: int = 5

    # Experiment settings
    seed: int = 42
    device: str = 'cuda'
    save_dir: str = './results'
    experiment_name: str = 'hcnet_experiment'


@dataclass
class NBodyConfig(TrainingConfig):
    """Configuration for N-body experiments."""
    n_train: int = 5000
    n_test: int = 1000
    n_particles: int = 5
    coord_dim: int = 3


@dataclass
class MD17Config(TrainingConfig):
    """Configuration for MD17 experiments."""
    molecule: str = 'ethanol'
    n_train: int = 1000
    n_test: int = 500
    cutoff: float = 5.0
    k_neighbors: int = 8


@dataclass
class ExperimentConfig:
    """Configuration for running multi-seed experiments."""
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    training_sizes: List[int] = field(default_factory=lambda: [100, 500, 1000, 5000])


def get_default_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()
