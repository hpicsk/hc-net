"""
Spinning N-body Dataset Generator.

Generates N-body systems where particles orbit a common center such that:
- Net velocity sum(v_i) ≈ 0 (vectors cancel due to symmetry)
- Net angular momentum L = sum(r_i × v_i) ≠ 0 (bivectors don't cancel)

This is the "killer experiment" dataset that demonstrates the vector
averaging collapse problem: standard vector mean-field loses all rotation
information, while Clifford multivector mean-field preserves it via bivectors.

Key properties:
1. Particles are placed in symmetric circular orbits
2. All particles rotate in the same direction (CCW or CW)
3. Due to symmetry, velocity vectors cancel: sum(v_i) ≈ 0
4. But angular momentum is preserved: sum(r_i × v_i) = N * |r| * |v| (all same sign)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class SpinningNBodyState:
    """State of a spinning N-body system."""
    positions: np.ndarray   # [N, 2]
    velocities: np.ndarray  # [N, 2]
    masses: np.ndarray      # [N]
    angular_momentum: float  # Total angular momentum L = sum(r × v)

    def to_array(self) -> np.ndarray:
        """Convert to [N, 4] array."""
        return np.concatenate([self.positions, self.velocities], axis=1)

    @staticmethod
    def from_array(arr: np.ndarray, masses: Optional[np.ndarray] = None) -> 'SpinningNBodyState':
        """Create from [N, 4] array."""
        n_particles = arr.shape[0]
        positions = arr[:, :2]
        velocities = arr[:, 2:]
        if masses is None:
            masses = np.ones(n_particles)
        # Compute angular momentum
        L = np.sum(positions[:, 0] * velocities[:, 1] - positions[:, 1] * velocities[:, 0])
        return SpinningNBodyState(positions, velocities, masses, L)


class SpinningNBodySimulator:
    """
    Simulator for spinning N-body systems.

    Creates systems where all particles orbit a common center of mass.
    This produces configurations where velocity vectors cancel but
    angular momentum is preserved.
    """

    def __init__(
        self,
        n_particles: int = 5,
        dt: float = 0.01,
        G: float = 1.0,
        softening: float = 0.5,
        central_mass: float = 10.0
    ):
        """
        Args:
            n_particles: Number of orbiting particles
            dt: Time step
            G: Gravitational constant
            softening: Softening parameter
            central_mass: Mass of central body (creates stable orbits)
        """
        self.n_particles = n_particles
        self.dt = dt
        self.G = G
        self.softening = softening
        self.central_mass = central_mass

    def initialize_spinning(
        self,
        radius_mean: float = 3.0,
        radius_std: float = 0.5,
        direction: int = 1,  # 1 = CCW, -1 = CW
        seed: Optional[int] = None
    ) -> SpinningNBodyState:
        """
        Initialize particles in circular orbits around origin.

        All particles rotate in the same direction, creating a system where:
        - Velocity vectors cancel (due to angular symmetry)
        - Angular momentum is non-zero (all rotations same direction)

        Args:
            radius_mean: Mean orbital radius
            radius_std: Standard deviation of radius
            direction: Rotation direction (1=CCW, -1=CW)
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)

        positions = np.zeros((self.n_particles, 2))
        velocities = np.zeros((self.n_particles, 2))
        masses = np.ones(self.n_particles)

        # Place particles evenly around a circle
        # This ensures velocity vectors cancel out
        for i in range(self.n_particles):
            # Even angular distribution
            angle = 2 * np.pi * i / self.n_particles

            # Slightly varied radius (but keep symmetry for velocity cancellation)
            radius = radius_mean + np.random.randn() * radius_std * 0.1

            # Position on circle
            positions[i] = [radius * np.cos(angle), radius * np.sin(angle)]

            # Orbital velocity: v = sqrt(G * M / r) for circular orbit
            orbital_speed = np.sqrt(self.G * self.central_mass / radius)

            # Velocity perpendicular to radius (tangent to circle)
            # direction controls CCW (+1) or CW (-1)
            velocities[i] = direction * orbital_speed * np.array([-np.sin(angle), np.cos(angle)])

        # Compute angular momentum: L = sum(x * vy - y * vx)
        L = np.sum(positions[:, 0] * velocities[:, 1] - positions[:, 1] * velocities[:, 0])

        return SpinningNBodyState(positions, velocities, masses, L)

    def compute_accelerations(self, state: SpinningNBodyState) -> np.ndarray:
        """
        Compute accelerations from central mass + inter-particle gravity.
        """
        N = state.positions.shape[0]
        accelerations = np.zeros((N, 2))

        # Central mass attraction (at origin)
        for i in range(N):
            r = state.positions[i]
            dist_sq = np.sum(r ** 2) + self.softening ** 2
            dist = np.sqrt(dist_sq)
            # Force toward origin
            accelerations[i] -= self.G * self.central_mass * r / (dist * dist_sq)

        # Inter-particle gravity
        for i in range(N):
            for j in range(N):
                if i != j:
                    r_ij = state.positions[j] - state.positions[i]
                    dist_sq = np.sum(r_ij ** 2) + self.softening ** 2
                    dist = np.sqrt(dist_sq)
                    accelerations[i] += self.G * state.masses[j] * r_ij / (dist * dist_sq)

        return accelerations

    def step(self, state: SpinningNBodyState) -> SpinningNBodyState:
        """Advance simulation by one time step using Velocity Verlet."""
        acc = self.compute_accelerations(state)

        # Update positions
        new_positions = (
            state.positions +
            state.velocities * self.dt +
            0.5 * acc * self.dt ** 2
        )

        # Get new accelerations
        temp_state = SpinningNBodyState(new_positions, state.velocities, state.masses, 0)
        new_acc = self.compute_accelerations(temp_state)

        # Update velocities
        new_velocities = state.velocities + 0.5 * (acc + new_acc) * self.dt

        # Compute new angular momentum
        L = np.sum(new_positions[:, 0] * new_velocities[:, 1] -
                   new_positions[:, 1] * new_velocities[:, 0])

        return SpinningNBodyState(new_positions, new_velocities, state.masses.copy(), L)


def generate_spinning_nbody_dataset(
    n_trajectories: int = 1000,
    n_particles: int = 5,
    prediction_horizon: int = 1,
    warmup_range: Tuple[int, int] = (10, 50),
    seed: int = 42,
    mixed_directions: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate spinning N-body prediction dataset.

    Args:
        n_trajectories: Number of samples
        n_particles: Particles per system
        prediction_horizon: Steps to predict ahead
        warmup_range: Range of warmup steps
        seed: Random seed
        mixed_directions: If True, randomly choose CW/CCW per sample

    Returns:
        inputs: [n_trajectories, n_particles, 4]
        targets: [n_trajectories, n_particles, 4]
        angular_momenta: [n_trajectories] - ground truth angular momentum
    """
    np.random.seed(seed)

    simulator = SpinningNBodySimulator(n_particles=n_particles)

    inputs = []
    targets = []
    angular_momenta = []

    for i in range(n_trajectories):
        # Direction: always CCW unless mixed
        direction = 1
        if mixed_directions:
            direction = np.random.choice([-1, 1])

        # Initialize spinning system
        state = simulator.initialize_spinning(
            radius_mean=3.0 + np.random.randn() * 0.5,
            direction=direction,
            seed=seed + i
        )

        # Warmup
        warmup = np.random.randint(warmup_range[0], warmup_range[1])
        for _ in range(warmup):
            state = simulator.step(state)

        # Record input
        inputs.append(state.to_array())
        angular_momenta.append(state.angular_momentum)

        # Simulate prediction_horizon steps
        for _ in range(prediction_horizon):
            state = simulator.step(state)

        # Record target
        targets.append(state.to_array())

    return (
        np.array(inputs, dtype=np.float32),
        np.array(targets, dtype=np.float32),
        np.array(angular_momenta, dtype=np.float32)
    )


class SpinningNBodyDataset(Dataset):
    """
    Spinning N-body dataset for testing vector averaging collapse.

    Key properties to verify:
    1. Average velocity is near zero: mean(v) ≈ 0
    2. Angular momentum is non-zero: L = sum(r × v) ≠ 0
    """

    def __init__(
        self,
        n_samples: int = 5000,
        n_particles: int = 5,
        prediction_horizon: int = 1,
        normalize: bool = True,
        seed: int = 42,
        mixed_directions: bool = False
    ):
        """
        Args:
            n_samples: Number of samples
            n_particles: Particles per system
            prediction_horizon: Steps to predict ahead
            normalize: If True, normalize inputs/outputs
            seed: Random seed
            mixed_directions: If True, mix CW and CCW rotations
        """
        self.n_particles = n_particles
        self.normalize = normalize

        # Generate data
        self.inputs, self.targets, self.angular_momenta = generate_spinning_nbody_dataset(
            n_trajectories=n_samples,
            n_particles=n_particles,
            prediction_horizon=prediction_horizon,
            seed=seed,
            mixed_directions=mixed_directions
        )

        # Verify the key property: velocities should nearly cancel
        avg_velocity = np.mean(self.inputs[:, :, 2:4], axis=1)  # [n_samples, 2]
        avg_vel_magnitude = np.linalg.norm(avg_velocity, axis=1).mean()
        particle_vel_magnitude = np.linalg.norm(self.inputs[:, :, 2:4], axis=2).mean()
        self.velocity_cancellation_ratio = avg_vel_magnitude / (particle_vel_magnitude + 1e-8)

        # Verify angular momentum is non-zero
        self.avg_angular_momentum = np.abs(self.angular_momenta).mean()

        # Normalize
        if normalize:
            self.input_mean = self.inputs.mean(axis=(0, 1), keepdims=True)
            self.input_std = self.inputs.std(axis=(0, 1), keepdims=True) + 1e-8
            self.target_mean = self.targets.mean(axis=(0, 1), keepdims=True)
            self.target_std = self.targets.std(axis=(0, 1), keepdims=True) + 1e-8

            self.inputs = (self.inputs - self.input_mean) / self.input_std
            self.targets = (self.targets - self.target_mean) / self.target_std

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )

    def get_statistics(self) -> Dict:
        """Return dataset statistics showing the key properties."""
        return {
            'velocity_cancellation_ratio': self.velocity_cancellation_ratio,
            'avg_angular_momentum': self.avg_angular_momentum,
            'note': 'Low velocity_cancellation_ratio + high angular_momentum = good spinning dataset'
        }


def get_spinning_nbody_loaders(
    n_train: int = 5000,
    n_test: int = 1000,
    n_particles: int = 5,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 42,
    mixed_directions: bool = False
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Get spinning N-body train/test dataloaders.

    Returns:
        (train_loader, test_loader, stats_dict)
    """
    train_dataset = SpinningNBodyDataset(
        n_samples=n_train,
        n_particles=n_particles,
        seed=seed,
        mixed_directions=mixed_directions
    )

    test_dataset = SpinningNBodyDataset(
        n_samples=n_test,
        n_particles=n_particles,
        seed=seed + 10000,
        mixed_directions=mixed_directions
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

    stats = {
        'train': train_dataset.get_statistics(),
        'test': test_dataset.get_statistics()
    }

    return train_loader, test_loader, stats


if __name__ == '__main__':
    print("Testing Spinning N-body Dataset...")
    print("=" * 60)

    # Generate dataset
    dataset = SpinningNBodyDataset(n_samples=1000, n_particles=5, seed=42)

    print(f"Dataset size: {len(dataset)}")
    print(f"Input shape: {dataset.inputs.shape}")
    print(f"Target shape: {dataset.targets.shape}")

    # Check key properties
    stats = dataset.get_statistics()
    print(f"\nKey Statistics:")
    print(f"  Velocity cancellation ratio: {stats['velocity_cancellation_ratio']:.4f}")
    print(f"  (Should be << 1, meaning velocities nearly cancel)")
    print(f"  Average angular momentum: {stats['avg_angular_momentum']:.4f}")
    print(f"  (Should be >> 0, meaning rotation is present)")

    # Verify with a single sample
    print(f"\nSingle sample analysis:")
    inp, tgt = dataset[0]
    print(f"  Input shape: {inp.shape}")
    print(f"  Target shape: {tgt.shape}")

    # Manually compute mean velocity for verification
    sample_unnorm = dataset.inputs[0] * dataset.input_std[0] + dataset.input_mean[0]
    velocities = sample_unnorm[:, 2:4]
    mean_vel = velocities.mean(axis=0)
    vel_magnitudes = np.linalg.norm(velocities, axis=1)

    print(f"  Mean velocity vector: [{mean_vel[0]:.4f}, {mean_vel[1]:.4f}]")
    print(f"  Mean velocity magnitude: {np.linalg.norm(mean_vel):.4f}")
    print(f"  Average particle velocity magnitude: {vel_magnitudes.mean():.4f}")
    print(f"  Angular momentum: {dataset.angular_momenta[0]:.4f}")

    # Test loaders
    train_loader, test_loader, stats = get_spinning_nbody_loaders(
        n_train=500, n_test=100, batch_size=32
    )
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    for batch_inp, batch_tgt in train_loader:
        print(f"Batch input shape: {batch_inp.shape}")
        print(f"Batch target shape: {batch_tgt.shape}")
        break

    print("\nSUCCESS: Spinning N-body dataset working!")
    print("=" * 60)
    print("\nThis dataset is designed to demonstrate vector averaging collapse:")
    print("- Mean velocity ≈ 0 (vector information lost)")
    print("- Angular momentum ≠ 0 (bivector information preserved)")
    print("- VectorMeanFieldNet should FAIL on this task")
    print("- HC-Net (with bivectors) should SUCCEED")
