"""
N-body Physics Prediction Dataset with Clifford Algebra Embedding.

Provides PyTorch datasets for training physics prediction models.
Supports both raw state vectors and Clifford algebra embeddings.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict
import os

# Support both package import and direct execution
try:
    from .nbody_simulation import (
        generate_nbody_dataset,
        rotate_state,
        NBodyState,
        NBodySimulator
    )
except ImportError:
    from importlib.util import spec_from_file_location, module_from_spec
    _DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
    _spec = spec_from_file_location(
        "nbody_simulation",
        os.path.join(_DATA_ROOT, 'nbody_simulation.py')
    )
    _nbody_sim = module_from_spec(_spec)
    _spec.loader.exec_module(_nbody_sim)
    generate_nbody_dataset = _nbody_sim.generate_nbody_dataset
    rotate_state = _nbody_sim.rotate_state
    NBodyState = _nbody_sim.NBodyState
    NBodySimulator = _nbody_sim.NBodySimulator


class NBodyDataset(Dataset):
    """
    N-body physics prediction dataset.

    Input: particle states (position + velocity)
    Target: particle states at t+1

    For Clifford mapping in Cl(4,0) or Cl(8,0):
    - Position (x, y) maps to vector: x*e1 + y*e2
    - Velocity (vx, vy) maps to vector: vx*e3 + vy*e4

    The geometric product will naturally create bivector interactions:
    - e1 e2: position "area" (rotation-related)
    - e1 e3, e2 e4: position-velocity cross terms
    - e3 e4: velocity "area"
    """

    def __init__(
        self,
        n_samples: int = 10000,
        n_particles: int = 5,
        prediction_horizon: int = 1,
        embed_clifford: bool = False,
        block_size: int = 8,
        normalize: bool = True,
        seed: int = 42
    ):
        """
        Args:
            n_samples: Number of trajectory samples
            n_particles: Particles per system
            prediction_horizon: Steps ahead to predict
            embed_clifford: If True, embed states in Clifford algebra
            block_size: Clifford algebra dimension for embedding
            normalize: If True, normalize inputs/outputs
            seed: Random seed
        """
        self.n_particles = n_particles
        self.prediction_horizon = prediction_horizon
        self.embed_clifford = embed_clifford
        self.block_size = block_size
        self.normalize = normalize
        self.seed = seed

        # Generate data
        self.inputs, self.targets = generate_nbody_dataset(
            n_trajectories=n_samples,
            n_particles=n_particles,
            prediction_horizon=prediction_horizon,
            seed=seed
        )

        # Compute normalization statistics
        if normalize:
            self.input_mean = self.inputs.mean(axis=(0, 1), keepdims=True)
            self.input_std = self.inputs.std(axis=(0, 1), keepdims=True) + 1e-8
            self.target_mean = self.targets.mean(axis=(0, 1), keepdims=True)
            self.target_std = self.targets.std(axis=(0, 1), keepdims=True) + 1e-8

            # Normalize
            self.inputs = (self.inputs - self.input_mean) / self.input_std
            self.targets = (self.targets - self.target_mean) / self.target_std

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_state = self.inputs[idx]  # [N, 4]
        target_state = self.targets[idx]  # [N, 4]

        if self.embed_clifford:
            input_state = self._embed_in_clifford(input_state)
            target_state = self._embed_in_clifford(target_state)

        return (
            torch.tensor(input_state, dtype=torch.float32),
            torch.tensor(target_state, dtype=torch.float32)
        )

    def _embed_in_clifford(self, state: np.ndarray) -> np.ndarray:
        """
        Embed particle state in Clifford algebra Cl(block_size, 0).

        state: [N, 4] with (x, y, vx, vy) per particle
        output: [N, 2^block_size] multivector per particle

        Embedding strategy for Cl(8,0) with block_size=8:
        - e1, e2: position basis (x, y)
        - e3, e4: velocity basis (vx, vy)
        - e5, e6, e7, e8: reserved (zeros)

        Basis element indexing: e_i has index 2^(i-1)
        - e1 = index 1
        - e2 = index 2
        - e3 = index 4
        - e4 = index 8
        """
        N = state.shape[0]
        mv_dim = 2 ** self.block_size
        embedded = np.zeros((N, mv_dim), dtype=np.float32)

        for i in range(N):
            x, y, vx, vy = state[i]

            # Place in vector components (indices 2^k for basis vectors)
            embedded[i, 1] = x    # e1 component
            embedded[i, 2] = y    # e2 component
            embedded[i, 4] = vx   # e3 component
            embedded[i, 8] = vy   # e4 component

        return embedded

    def get_normalization_params(self) -> Dict:
        """Return normalization parameters for denormalization."""
        if self.normalize:
            return {
                'input_mean': self.input_mean,
                'input_std': self.input_std,
                'target_mean': self.target_mean,
                'target_std': self.target_std
            }
        return {}


class RotatedNBodyDataset(Dataset):
    """
    OOD test set: N-body systems with systematic rotation.

    Used to test rotation equivariance/invariance of PCNN.
    Takes a base dataset and applies rotations to all samples.
    """

    def __init__(
        self,
        n_samples: int = 2000,
        n_particles: int = 5,
        rotation_angles: List[float] = [45, 90],
        embed_clifford: bool = False,
        block_size: int = 8,
        seed: int = 42
    ):
        """
        Args:
            n_samples: Number of base samples (will be multiplied by num angles)
            n_particles: Particles per system
            rotation_angles: Rotation angles to apply (degrees)
            embed_clifford: If True, embed states in Clifford algebra
            block_size: Clifford algebra dimension
            seed: Random seed
        """
        self.n_particles = n_particles
        self.rotation_angles = rotation_angles
        self.embed_clifford = embed_clifford
        self.block_size = block_size

        # Generate base dataset
        self.base_inputs, self.base_targets = generate_nbody_dataset(
            n_trajectories=n_samples,
            n_particles=n_particles,
            prediction_horizon=1,
            seed=seed + 10000  # Different seed from training
        )

    def __len__(self) -> int:
        return len(self.base_inputs) * len(self.rotation_angles)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        base_idx = idx // len(self.rotation_angles)
        angle_idx = idx % len(self.rotation_angles)
        angle = self.rotation_angles[angle_idx]

        # Get base sample
        input_state = self.base_inputs[base_idx]
        target_state = self.base_targets[base_idx]

        # Apply rotation to both input and target
        input_rotated = self._rotate_array(input_state, angle)
        target_rotated = self._rotate_array(target_state, angle)

        if self.embed_clifford:
            input_rotated = self._embed_in_clifford(input_rotated)
            target_rotated = self._embed_in_clifford(target_rotated)

        return (
            torch.tensor(input_rotated, dtype=torch.float32),
            torch.tensor(target_rotated, dtype=torch.float32),
            angle
        )

    def _rotate_array(self, state: np.ndarray, angle: float) -> np.ndarray:
        """Rotate state array by angle degrees."""
        angle_rad = np.radians(angle)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[c, -s], [s, c]])

        # Rotate positions (columns 0-1)
        positions = state[:, :2] @ rotation_matrix.T
        # Rotate velocities (columns 2-3)
        velocities = state[:, 2:] @ rotation_matrix.T

        return np.concatenate([positions, velocities], axis=1).astype(np.float32)

    def _embed_in_clifford(self, state: np.ndarray) -> np.ndarray:
        """Same as NBodyDataset embedding."""
        N = state.shape[0]
        mv_dim = 2 ** self.block_size
        embedded = np.zeros((N, mv_dim), dtype=np.float32)

        for i in range(N):
            x, y, vx, vy = state[i]
            embedded[i, 1] = x
            embedded[i, 2] = y
            embedded[i, 4] = vx
            embedded[i, 8] = vy

        return embedded


def get_nbody_loaders(
    n_train: int = 10000,
    n_test: int = 2000,
    n_particles: int = 5,
    batch_size: int = 128,
    embed_clifford: bool = False,
    block_size: int = 8,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Get N-body train/test dataloaders.

    Args:
        n_train: Number of training samples
        n_test: Number of test samples
        n_particles: Particles per system
        batch_size: Batch size
        embed_clifford: Whether to embed in Clifford algebra
        block_size: Clifford algebra dimension
        num_workers: Data loading workers
        seed: Random seed

    Returns:
        (train_loader, test_loader)
    """
    train_dataset = NBodyDataset(
        n_samples=n_train,
        n_particles=n_particles,
        embed_clifford=embed_clifford,
        block_size=block_size,
        seed=seed
    )

    test_dataset = NBodyDataset(
        n_samples=n_test,
        n_particles=n_particles,
        embed_clifford=embed_clifford,
        block_size=block_size,
        seed=seed + 10000
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def get_nbody_loaders_with_ood(
    n_train: int = 10000,
    n_test: int = 2000,
    n_particles: int = 5,
    rotation_angles: List[float] = [0, 45, 90],
    batch_size: int = 128,
    embed_clifford: bool = False,
    block_size: int = 8,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict[float, DataLoader]]:
    """
    Get N-body dataloaders including OOD rotated test sets.

    Args:
        n_train: Number of training samples
        n_test: Number of test samples
        n_particles: Particles per system
        rotation_angles: Rotation angles for OOD test sets
        batch_size: Batch size
        embed_clifford: Whether to embed in Clifford algebra
        block_size: Clifford algebra dimension
        num_workers: Data loading workers
        seed: Random seed

    Returns:
        (train_loader, test_loader, ood_loaders_dict)
        where ood_loaders_dict maps angle -> DataLoader
    """
    train_loader, test_loader = get_nbody_loaders(
        n_train=n_train,
        n_test=n_test,
        n_particles=n_particles,
        batch_size=batch_size,
        embed_clifford=embed_clifford,
        block_size=block_size,
        num_workers=num_workers,
        seed=seed
    )

    # Create OOD loaders for each angle
    ood_loaders = {}
    for angle in rotation_angles:
        if angle == 0:
            # 0-degree rotation is same as IID test
            ood_loaders[angle] = test_loader
        else:
            ood_dataset = RotatedNBodyDataset(
                n_samples=n_test,
                n_particles=n_particles,
                rotation_angles=[angle],
                embed_clifford=embed_clifford,
                block_size=block_size,
                seed=seed
            )
            ood_loaders[angle] = DataLoader(
                ood_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

    return train_loader, test_loader, ood_loaders


# =============================================================================
# 3D N-body Dataset Classes
# =============================================================================

# Import 3D simulation
try:
    from .nbody_simulation_3d import (
        generate_nbody_dataset_3d,
        rotate_state_3d,
        NBodyState3D,
        NBodySimulator3D
    )
except ImportError:
    _spec_3d = spec_from_file_location(
        "nbody_simulation_3d",
        os.path.join(_DATA_ROOT, 'nbody_simulation_3d.py')
    )
    _nbody_sim_3d = module_from_spec(_spec_3d)
    _spec_3d.loader.exec_module(_nbody_sim_3d)
    generate_nbody_dataset_3d = _nbody_sim_3d.generate_nbody_dataset_3d
    rotate_state_3d = _nbody_sim_3d.rotate_state_3d
    NBodyState3D = _nbody_sim_3d.NBodyState3D
    NBodySimulator3D = _nbody_sim_3d.NBodySimulator3D


class NBodyDataset3D(Dataset):
    """
    3D N-body physics prediction dataset.

    Input: particle states (position + velocity) [N, 6]
    Target: particle states at t+1 [N, 6]
    """

    def __init__(
        self,
        n_samples: int = 10000,
        n_particles: int = 5,
        seed: int = 42
    ):
        """
        Args:
            n_samples: Number of trajectory samples
            n_particles: Particles per system
            seed: Random seed
        """
        self.n_samples = n_samples
        self.n_particles = n_particles

        # Generate dataset
        self.inputs, self.targets = generate_nbody_dataset_3d(
            n_trajectories=n_samples,
            n_particles=n_particles,
            seed=seed
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )


class RotatedNBodyDataset3D(Dataset):
    """
    3D N-body dataset with rotations applied for OOD testing.

    Supports rotation around arbitrary axes or Euler angles.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_particles: int = 5,
        rotation_angles: List[Tuple[float, float, float]] = None,
        seed: int = 42
    ):
        """
        Args:
            n_samples: Number of samples
            n_particles: Particles per system
            rotation_angles: List of (angle_z, angle_y, angle_x) Euler angles
            seed: Random seed
        """
        self.n_samples = n_samples
        self.n_particles = n_particles

        if rotation_angles is None:
            rotation_angles = [(45, 0, 0), (0, 45, 0), (0, 0, 45), (90, 0, 0)]

        self.rotation_angles = rotation_angles

        # Generate base dataset
        inputs_base, targets_base = generate_nbody_dataset_3d(
            n_trajectories=n_samples,
            n_particles=n_particles,
            seed=seed
        )

        # Apply rotations
        np.random.seed(seed + 1000)
        self.inputs = []
        self.targets = []
        self.applied_angles = []

        for i in range(n_samples):
            # Pick rotation
            rot_idx = i % len(rotation_angles)
            angles = rotation_angles[rot_idx]

            # Apply rotation
            state_in = NBodyState3D.from_array(inputs_base[i])
            state_target = NBodyState3D.from_array(targets_base[i])

            state_in_rot = rotate_state_3d(state_in, angles)
            state_target_rot = rotate_state_3d(state_target, angles)

            self.inputs.append(state_in_rot.to_array())
            self.targets.append(state_target_rot.to_array())
            self.applied_angles.append(angles)

        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            self.applied_angles[idx]
        )


def get_nbody_loaders_3d(
    n_train: int = 10000,
    n_test: int = 2000,
    n_particles: int = 5,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """Get 3D N-body train and test dataloaders."""
    train_dataset = NBodyDataset3D(
        n_samples=n_train,
        n_particles=n_particles,
        seed=seed
    )

    test_dataset = NBodyDataset3D(
        n_samples=n_test,
        n_particles=n_particles,
        seed=seed + 10000
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def get_nbody_loaders_3d_with_ood(
    n_train: int = 10000,
    n_test: int = 2000,
    n_particles: int = 5,
    rotation_angles: List[Tuple[float, float, float]] = None,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict[Tuple, DataLoader]]:
    """
    Get 3D N-body dataloaders including OOD rotated test sets.

    Args:
        n_train: Number of training samples
        n_test: Number of test samples
        n_particles: Particles per system
        rotation_angles: List of (angle_z, angle_y, angle_x) tuples for OOD
        batch_size: Batch size
        num_workers: Data loading workers
        seed: Random seed

    Returns:
        (train_loader, test_loader, ood_loaders_dict)
        where ood_loaders_dict maps (angle_z, angle_y, angle_x) -> DataLoader
    """
    if rotation_angles is None:
        rotation_angles = [
            (0, 0, 0),      # Identity
            (45, 0, 0),     # 45 around z
            (0, 45, 0),     # 45 around y
            (0, 0, 45),     # 45 around x
            (90, 0, 0),     # 90 around z
            (0, 90, 0),     # 90 around y
            (0, 0, 90),     # 90 around x
        ]

    train_loader, test_loader = get_nbody_loaders_3d(
        n_train=n_train,
        n_test=n_test,
        n_particles=n_particles,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed
    )

    # Create OOD loaders for each rotation
    ood_loaders = {}
    for angles in rotation_angles:
        if angles == (0, 0, 0):
            ood_loaders[angles] = test_loader
        else:
            ood_dataset = RotatedNBodyDataset3D(
                n_samples=n_test,
                n_particles=n_particles,
                rotation_angles=[angles],
                seed=seed
            )
            ood_loaders[angles] = DataLoader(
                ood_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

    return train_loader, test_loader, ood_loaders


if __name__ == '__main__':
    # Test the dataset
    print("Testing N-body Dataset...")

    # Test basic dataset
    dataset = NBodyDataset(n_samples=100, n_particles=5, embed_clifford=False)
    print(f"Dataset size: {len(dataset)}")

    input_state, target_state = dataset[0]
    print(f"Raw input shape: {input_state.shape}")
    print(f"Raw target shape: {target_state.shape}")

    # Test Clifford embedding
    dataset_clifford = NBodyDataset(n_samples=100, n_particles=5, embed_clifford=True, block_size=8)
    input_mv, target_mv = dataset_clifford[0]
    print(f"Clifford input shape: {input_mv.shape}")
    print(f"Non-zero Clifford components: {(input_mv != 0).sum().item()}")

    # Test OOD dataset
    ood_dataset = RotatedNBodyDataset(n_samples=50, n_particles=5, rotation_angles=[45, 90])
    print(f"OOD dataset size: {len(ood_dataset)}")
    inp, tgt, angle = ood_dataset[0]
    print(f"OOD sample angle: {angle}")

    # Test loaders
    train_loader, test_loader, ood_loaders = get_nbody_loaders_with_ood(
        n_train=500, n_test=100, batch_size=32,
        rotation_angles=[0, 45, 90]
    )
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"OOD angles: {list(ood_loaders.keys())}")

    # Test a batch
    for batch_inp, batch_tgt in train_loader:
        print(f"Batch input shape: {batch_inp.shape}")
        print(f"Batch target shape: {batch_tgt.shape}")
        break

    print("\nSUCCESS: N-body dataset working!")
