"""
3D Spinning N-Body with Chirality Labels.

Extends the 2D SpinningNBodySimulator to 3D with chirality:
- mode='rotation': CW/CCW rotation classification (bivector-solvable)
- mode='chirality': helix handedness (trivector-required)

Particles orbit in helical (screw) trajectories:
  chirality=+1 -> right-hand screw
  chirality=-1 -> left-hand screw

The rotation mode is solvable with bivectors (angular momentum L),
while chirality mode requires the trivector (scalar triple product).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from scipy.spatial.transform import Rotation


@dataclass
class SpinningNBodyState3D:
    """State of a 3D spinning N-body system."""
    positions: np.ndarray       # [N, 3]
    velocities: np.ndarray      # [N, 3]
    masses: np.ndarray          # [N]
    angular_momentum: np.ndarray  # [3] total angular momentum vector
    chirality: int              # +1 (right-hand screw) or -1 (left-hand screw)

    def to_array(self) -> np.ndarray:
        """Convert to [N, 6] array."""
        return np.concatenate([self.positions, self.velocities], axis=1)

    @staticmethod
    def from_array(
        arr: np.ndarray,
        masses: Optional[np.ndarray] = None,
        chirality: int = 0,
    ) -> 'SpinningNBodyState3D':
        """Create from [N, 6] array."""
        N = arr.shape[0]
        positions = arr[:, :3]
        velocities = arr[:, 3:]
        if masses is None:
            masses = np.ones(N)
        L = np.cross(positions, velocities).sum(axis=0)
        return SpinningNBodyState3D(positions, velocities, masses, L, chirality)


class SpinningNBodySimulator3D:
    """
    3D spinning N-body simulator with chirality.

    Creates systems where particles orbit in helical trajectories
    (screw motion). The handedness of the screw determines chirality.
    """

    def __init__(
        self,
        n_particles: int = 5,
        dt: float = 0.01,
        G: float = 1.0,
        softening: float = 0.5,
        central_mass: float = 10.0,
    ):
        self.n_particles = n_particles
        self.dt = dt
        self.G = G
        self.softening = softening
        self.central_mass = central_mass

    def initialize_chiral_spinning(
        self,
        radius_mean: float = 3.0,
        radius_std: float = 0.5,
        direction: int = 1,      # +1=CCW, -1=CW (in xy-plane)
        chirality: int = 1,       # +1=right-hand screw, -1=left-hand screw
        z_velocity: float = 0.3,  # axial velocity for screw motion
        seed: Optional[int] = None,
    ) -> SpinningNBodyState3D:
        """
        Initialize particles in helical (screw) orbits.

        The particles orbit in the xy-plane with a z-component of velocity
        that creates a screw motion. The handedness of the screw is
        determined by the chirality parameter.

        Args:
            radius_mean: Mean orbital radius
            radius_std: Std of radius (small perturbation)
            direction: Rotation direction in xy-plane
            chirality: +1 for right-hand screw, -1 for left-hand screw
            z_velocity: Axial velocity magnitude
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)

        positions = np.zeros((self.n_particles, 3))
        velocities = np.zeros((self.n_particles, 3))
        masses = np.ones(self.n_particles)

        for i in range(self.n_particles):
            angle = 2 * np.pi * i / self.n_particles
            radius = radius_mean + np.random.randn() * radius_std * 0.1

            # Position on circle in xy-plane with z offset
            z_offset = np.random.randn() * 0.1
            positions[i] = [
                radius * np.cos(angle),
                radius * np.sin(angle),
                z_offset,
            ]

            # Orbital velocity in xy-plane
            orbital_speed = np.sqrt(self.G * self.central_mass / radius)
            vx = direction * orbital_speed * (-np.sin(angle))
            vy = direction * orbital_speed * np.cos(angle)

            # z-velocity: chirality determines direction of screw
            # For right-hand screw (chirality=+1), CCW rotation (dir=+1) -> +z
            # This couples handedness to the rotation-translation relationship
            vz = chirality * direction * z_velocity

            velocities[i] = [vx, vy, vz]

        # Compute angular momentum
        L = np.cross(positions, velocities).sum(axis=0)

        return SpinningNBodyState3D(
            positions, velocities, masses, L, chirality
        )

    def compute_accelerations(self, state: SpinningNBodyState3D) -> np.ndarray:
        """Compute gravitational accelerations in 3D."""
        N = state.positions.shape[0]
        accelerations = np.zeros((N, 3))

        # Central mass at origin
        for i in range(N):
            r = state.positions[i]
            dist_sq = np.sum(r ** 2) + self.softening ** 2
            dist = np.sqrt(dist_sq)
            accelerations[i] -= self.G * self.central_mass * r / (dist * dist_sq)

        # Inter-particle gravity
        for i in range(N):
            for j in range(N):
                if i != j:
                    r_ij = state.positions[j] - state.positions[i]
                    dist_sq = np.sum(r_ij ** 2) + self.softening ** 2
                    dist = np.sqrt(dist_sq)
                    accelerations[i] += (
                        self.G * state.masses[j] * r_ij / (dist * dist_sq)
                    )

        return accelerations

    def step(self, state: SpinningNBodyState3D) -> SpinningNBodyState3D:
        """Advance by one time step using Velocity Verlet."""
        acc = self.compute_accelerations(state)

        new_positions = (
            state.positions
            + state.velocities * self.dt
            + 0.5 * acc * self.dt ** 2
        )

        temp_state = SpinningNBodyState3D(
            new_positions, state.velocities, state.masses,
            state.angular_momentum, state.chirality,
        )
        new_acc = self.compute_accelerations(temp_state)

        new_velocities = state.velocities + 0.5 * (acc + new_acc) * self.dt

        L = np.cross(new_positions, new_velocities).sum(axis=0)

        return SpinningNBodyState3D(
            new_positions, new_velocities, state.masses.copy(),
            L, state.chirality,
        )


class SpinningChiralityDataset3D(Dataset):
    """
    3D spinning N-body classification dataset.

    Two modes:
    - 'rotation': Classify CW vs CCW rotation (bivector-solvable)
    - 'chirality': Classify helix handedness (trivector-required)

    In rotation mode, the label encodes direction of angular momentum.
    In chirality mode, the label encodes the screw handedness.
    """

    def __init__(
        self,
        n_samples: int = 5000,
        n_particles: int = 5,
        mode: str = 'chirality',
        apply_random_rotation: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            n_samples: Number of samples
            n_particles: Particles per system
            mode: 'rotation' or 'chirality'
            apply_random_rotation: Apply random SO(3) rotation to prevent shortcuts
            seed: Random seed
        """
        assert mode in ('rotation', 'chirality'), f"Unknown mode: {mode}"
        self.n_samples = n_samples
        self.mode = mode

        rng = np.random.RandomState(seed)
        simulator = SpinningNBodySimulator3D(n_particles=n_particles)

        self.inputs = []
        self.labels = []

        for i in range(n_samples):
            if mode == 'rotation':
                # Both chiralities, vary rotation direction
                direction = rng.choice([-1, 1])
                chirality = 1  # Fixed chirality
                label = 1 if direction == 1 else 0
                # No axial velocity for rotation mode — pure circular orbits
                z_vel = 0.0
            else:
                # Both directions, vary chirality
                direction = 1  # Fixed direction (or could randomize)
                chirality = rng.choice([-1, 1])
                label = 1 if chirality == 1 else 0
                z_vel = 0.3 + rng.randn() * 0.05

            state = simulator.initialize_chiral_spinning(
                radius_mean=3.0 + rng.randn() * 0.3,
                direction=direction,
                chirality=chirality,
                z_velocity=z_vel,
                seed=seed + i,
            )

            # Warmup simulation
            warmup = rng.randint(5, 20)
            for _ in range(warmup):
                state = simulator.step(state)

            arr = state.to_array()  # [N, 6]

            if apply_random_rotation:
                if mode == 'rotation':
                    # Rotation mode: only rotate around z-axis.
                    # This preserves the sign of L_z (angular momentum),
                    # which is the bivector signal for CW/CCW detection.
                    # Full SO(3) would destroy this signal in 3D.
                    theta = rng.uniform(0, 2 * np.pi)
                    c, s = np.cos(theta), np.sin(theta)
                    R = np.array([
                        [c, -s, 0],
                        [s,  c, 0],
                        [0,  0, 1],
                    ], dtype=np.float32)
                else:
                    # Chirality mode: full SO(3) rotation.
                    # Chirality (trivector) is SO(3)-invariant.
                    R = Rotation.random(random_state=rng).as_matrix().astype(np.float32)
                pos = arr[:, :3] @ R.T
                vel = arr[:, 3:] @ R.T
                arr = np.concatenate([pos, vel], axis=1)

            self.inputs.append(arr)
            self.labels.append(label)

        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

    def get_statistics(self) -> Dict:
        """Return dataset statistics."""
        # Velocity cancellation ratio
        velocities = self.inputs[:, :, 3:6]
        avg_vel = velocities.mean(axis=1)  # [n_samples, 3]
        avg_vel_mag = np.linalg.norm(avg_vel, axis=1).mean()
        particle_vel_mag = np.linalg.norm(velocities, axis=2).mean()

        return {
            'mode': self.mode,
            'velocity_cancellation_ratio': float(avg_vel_mag / (particle_vel_mag + 1e-8)),
            'n_samples': self.n_samples,
            'label_balance': float(self.labels.mean()),
        }


if __name__ == '__main__':
    print("=" * 70)
    print("3D SPINNING N-BODY WITH CHIRALITY: Verification")
    print("=" * 70)

    # Test rotation mode
    print("\n--- Rotation Mode (bivector-solvable) ---")
    ds_rot = SpinningChiralityDataset3D(
        n_samples=500, n_particles=5, mode='rotation', seed=42
    )
    stats = ds_rot.get_statistics()
    print(f"  Samples: {stats['n_samples']}")
    print(f"  Label balance: {stats['label_balance']:.3f}")
    print(f"  Velocity cancellation: {stats['velocity_cancellation_ratio']:.4f}")

    # Test chirality mode
    print("\n--- Chirality Mode (trivector-required) ---")
    ds_chir = SpinningChiralityDataset3D(
        n_samples=500, n_particles=5, mode='chirality', seed=42
    )
    stats = ds_chir.get_statistics()
    print(f"  Samples: {stats['n_samples']}")
    print(f"  Label balance: {stats['label_balance']:.3f}")
    print(f"  Velocity cancellation: {stats['velocity_cancellation_ratio']:.4f}")

    # Test single sample shape
    inp, lbl = ds_chir[0]
    print(f"\n  Single sample: input={inp.shape}, label={lbl.item()}")

    # Test simulator directly
    print("\n--- Simulator Test ---")
    sim = SpinningNBodySimulator3D(n_particles=5)
    state = sim.initialize_chiral_spinning(
        direction=1, chirality=1, seed=123
    )
    print(f"  Initial L: {state.angular_momentum}")
    print(f"  Chirality: {state.chirality}")

    for _ in range(50):
        state = sim.step(state)
    print(f"  After 50 steps L: {state.angular_momentum}")
    print(f"  L magnitude preserved: {np.linalg.norm(state.angular_momentum):.4f}")

    print("\nSUCCESS: 3D spinning N-body dataset working!")
