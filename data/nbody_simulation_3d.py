"""
3D N-body Gravitational Simulation.

Extended from 2D version for comparison experiments.
Uses Velocity Verlet integration for accurate physics in 3D.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class NBodyState3D:
    """State of 3D N-body system."""
    positions: np.ndarray   # [N, 3] x, y, z positions
    velocities: np.ndarray  # [N, 3] vx, vy, vz velocities
    masses: np.ndarray      # [N] particle masses

    def copy(self) -> 'NBodyState3D':
        """Create a deep copy of the state."""
        return NBodyState3D(
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            masses=self.masses.copy()
        )

    def to_array(self) -> np.ndarray:
        """Convert to flat array [N, 6] with (x, y, z, vx, vy, vz) per particle."""
        return np.concatenate([self.positions, self.velocities], axis=1)

    @staticmethod
    def from_array(arr: np.ndarray, masses: Optional[np.ndarray] = None) -> 'NBodyState3D':
        """Create from flat array [N, 6]."""
        n_particles = arr.shape[0]
        positions = arr[:, :3]
        velocities = arr[:, 3:]
        if masses is None:
            masses = np.ones(n_particles)
        return NBodyState3D(positions, velocities, masses)


class NBodySimulator3D:
    """
    3D gravitational N-body simulator using Velocity Verlet integration.

    The simulator models gravitational attraction between particles:
        F_ij = G * m_i * m_j * (r_j - r_i) / |r_j - r_i|^3

    Uses softening to avoid singularities when particles get close.
    """

    def __init__(
        self,
        n_particles: int = 5,
        box_size: float = 10.0,
        dt: float = 0.01,
        G: float = 1.0,
        softening: float = 0.5
    ):
        """
        Args:
            n_particles: Number of particles in the system
            box_size: Size of the simulation box (particles initialized within)
            dt: Time step for integration
            G: Gravitational constant
            softening: Softening parameter to avoid singularities
        """
        self.n_particles = n_particles
        self.box_size = box_size
        self.dt = dt
        self.G = G
        self.softening = softening

    def initialize_random(self, seed: Optional[int] = None) -> NBodyState3D:
        """
        Initialize random particle configuration in 3D.

        Particles are placed randomly in the box with small random velocities.
        """
        if seed is not None:
            np.random.seed(seed)

        # Random positions within box
        positions = np.random.uniform(
            -self.box_size / 2,
            self.box_size / 2,
            (self.n_particles, 3)
        )

        # Random velocities (relatively small)
        velocities = np.random.randn(self.n_particles, 3) * 0.5

        # Unit masses for simplicity
        masses = np.ones(self.n_particles)

        return NBodyState3D(positions, velocities, masses)

    def initialize_orbit(self, seed: Optional[int] = None) -> NBodyState3D:
        """
        Initialize particles in approximate orbits in 3D.

        Creates a more stable configuration with orbits in xy plane
        with slight z-axis perturbations.
        """
        if seed is not None:
            np.random.seed(seed)

        positions = np.zeros((self.n_particles, 3))
        velocities = np.zeros((self.n_particles, 3))
        masses = np.ones(self.n_particles)

        # Central heavy mass
        masses[0] = 5.0

        # Other particles in approximate orbits
        for i in range(1, self.n_particles):
            angle = 2 * np.pi * i / (self.n_particles - 1) + np.random.randn() * 0.1
            radius = 2.0 + np.random.randn() * 0.5
            z_offset = np.random.randn() * 0.3  # Small z displacement

            positions[i] = [
                radius * np.cos(angle),
                radius * np.sin(angle),
                z_offset
            ]

            # Circular orbit velocity in xy plane: v = sqrt(GM/r)
            orbital_speed = np.sqrt(self.G * masses[0] / radius)
            # Perpendicular to radius in xy plane, with small z component
            velocities[i] = [
                -orbital_speed * np.sin(angle),
                orbital_speed * np.cos(angle),
                np.random.randn() * 0.1
            ]

        return NBodyState3D(positions, velocities, masses)

    def compute_accelerations_vectorized(self, state: NBodyState3D) -> np.ndarray:
        """Vectorized computation of gravitational accelerations in 3D."""
        N = state.positions.shape[0]

        # Pairwise position differences: [N, N, 3]
        r_ij = state.positions[np.newaxis, :, :] - state.positions[:, np.newaxis, :]

        # Pairwise distances squared: [N, N]
        dist_sq = np.sum(r_ij ** 2, axis=2) + self.softening ** 2

        # Avoid self-interaction
        np.fill_diagonal(dist_sq, np.inf)

        # Distance: [N, N]
        dist = np.sqrt(dist_sq)

        # Force magnitude factor: [N, N]
        force_factor = self.G * state.masses[np.newaxis, :] / (dist * dist_sq)

        # Accelerations: sum over j
        accelerations = np.sum(force_factor[:, :, np.newaxis] * r_ij, axis=1)

        return accelerations

    def step(self, state: NBodyState3D) -> NBodyState3D:
        """
        Advance simulation by one time step using Velocity Verlet.
        """
        # Get current accelerations
        acc = self.compute_accelerations_vectorized(state)

        # Update positions: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
        new_positions = (
            state.positions +
            state.velocities * self.dt +
            0.5 * acc * self.dt ** 2
        )

        # Create intermediate state for new accelerations
        temp_state = NBodyState3D(new_positions, state.velocities, state.masses)
        new_acc = self.compute_accelerations_vectorized(temp_state)

        # Update velocities: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        new_velocities = state.velocities + 0.5 * (acc + new_acc) * self.dt

        return NBodyState3D(new_positions, new_velocities, state.masses.copy())

    def generate_trajectory(
        self,
        initial_state: NBodyState3D,
        n_steps: int
    ) -> List[NBodyState3D]:
        """Generate trajectory of n_steps from initial state."""
        trajectory = [initial_state.copy()]
        state = initial_state

        for _ in range(n_steps):
            state = self.step(state)
            trajectory.append(state.copy())

        return trajectory

    def compute_energy(self, state: NBodyState3D) -> Tuple[float, float, float]:
        """Compute total energy of the system."""
        # Kinetic energy: 0.5 * sum(m * v^2)
        kinetic = 0.5 * np.sum(state.masses[:, np.newaxis] * state.velocities ** 2)

        # Potential energy: -G * sum_{i<j} m_i * m_j / r_ij
        potential = 0.0
        N = state.positions.shape[0]
        for i in range(N):
            for j in range(i + 1, N):
                r_ij = state.positions[j] - state.positions[i]
                dist = np.sqrt(np.sum(r_ij ** 2) + self.softening ** 2)
                potential -= self.G * state.masses[i] * state.masses[j] / dist

        return kinetic, potential, kinetic + potential


def rotate_state_3d(
    state: NBodyState3D,
    angles_degrees: Tuple[float, float, float]
) -> NBodyState3D:
    """
    Rotate entire 3D system by given Euler angles (ZYX convention).

    Args:
        state: Current state
        angles_degrees: (angle_z, angle_y, angle_x) in degrees

    Returns:
        Rotated state
    """
    alpha, beta, gamma = np.radians(angles_degrees)

    # Rotation matrix (ZYX Euler angles)
    Rz = np.array([
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(gamma), -np.sin(gamma)],
        [0, np.sin(gamma), np.cos(gamma)]
    ])

    R = Rz @ Ry @ Rx

    new_positions = state.positions @ R.T
    new_velocities = state.velocities @ R.T

    return NBodyState3D(new_positions, new_velocities, state.masses.copy())


def rotate_state_3d_axis_angle(
    state: NBodyState3D,
    axis: np.ndarray,
    angle_degrees: float
) -> NBodyState3D:
    """
    Rotate entire 3D system around given axis by angle.

    Uses Rodrigues' rotation formula.

    Args:
        state: Current state
        axis: Unit vector for rotation axis [3]
        angle_degrees: Rotation angle in degrees

    Returns:
        Rotated state
    """
    angle = np.radians(angle_degrees)
    axis = axis / np.linalg.norm(axis)

    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    new_positions = state.positions @ R.T
    new_velocities = state.velocities @ R.T

    return NBodyState3D(new_positions, new_velocities, state.masses.copy())


def generate_nbody_dataset_3d(
    n_trajectories: int = 1000,
    n_particles: int = 5,
    prediction_horizon: int = 1,
    warmup_range: Tuple[int, int] = (10, 50),
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 3D N-body prediction dataset.

    Each sample is a pair (input_state, target_state) where:
    - input_state: particle states at time t [N, 6] (x, y, z, vx, vy, vz)
    - target_state: particle states at time t + prediction_horizon

    Args:
        n_trajectories: Number of independent trajectory samples
        n_particles: Particles per system
        prediction_horizon: Steps to predict ahead
        warmup_range: Range of warmup steps to get varied configurations
        seed: Random seed

    Returns:
        inputs: [n_trajectories, n_particles, 6]
        targets: [n_trajectories, n_particles, 6]
    """
    np.random.seed(seed)

    simulator = NBodySimulator3D(n_particles=n_particles)

    inputs = []
    targets = []

    for i in range(n_trajectories):
        # Initialize random system
        state = simulator.initialize_random(seed=seed + i)

        # Simulate for random warmup steps to get varied configurations
        warmup = np.random.randint(warmup_range[0], warmup_range[1])
        for _ in range(warmup):
            state = simulator.step(state)

        # Record input state
        inputs.append(state.to_array())

        # Simulate prediction_horizon steps
        for _ in range(prediction_horizon):
            state = simulator.step(state)

        # Record target state
        targets.append(state.to_array())

    return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)


def generate_nbody_dataset_3d_rotated(
    n_trajectories: int = 1000,
    n_particles: int = 5,
    rotation_angles: List[Tuple[float, float, float]] = None,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 3D dataset with rotation applied for OOD testing.

    Args:
        n_trajectories: Number of samples
        n_particles: Particles per system
        rotation_angles: List of (angle_z, angle_y, angle_x) tuples
        seed: Random seed

    Returns:
        inputs: [n_trajectories, n_particles, 6]
        targets: [n_trajectories, n_particles, 6]
        rotation_indices: [n_trajectories] index of rotation applied
    """
    if rotation_angles is None:
        rotation_angles = [
            (0, 0, 0),      # Identity
            (45, 0, 0),     # 45 around z
            (0, 45, 0),     # 45 around y
            (0, 0, 45),     # 45 around x
            (45, 45, 0),    # Combined
            (90, 0, 0),     # 90 around z
            (0, 90, 0),     # 90 around y
            (0, 0, 90),     # 90 around x
        ]

    # First generate base dataset
    inputs_base, targets_base = generate_nbody_dataset_3d(
        n_trajectories=n_trajectories,
        n_particles=n_particles,
        seed=seed
    )

    inputs = []
    targets = []
    rotation_indices = []

    np.random.seed(seed + 1000)

    for i in range(n_trajectories):
        # Pick random rotation
        rot_idx = np.random.randint(len(rotation_angles))
        angles = rotation_angles[rot_idx]

        # Create state from array
        state_in = NBodyState3D.from_array(inputs_base[i])
        state_target = NBodyState3D.from_array(targets_base[i])

        # Apply rotation
        state_in_rot = rotate_state_3d(state_in, angles)
        state_target_rot = rotate_state_3d(state_target, angles)

        inputs.append(state_in_rot.to_array())
        targets.append(state_target_rot.to_array())
        rotation_indices.append(rot_idx)

    return (
        np.array(inputs, dtype=np.float32),
        np.array(targets, dtype=np.float32),
        np.array(rotation_indices, dtype=np.int32)
    )


if __name__ == '__main__':
    # Test the 3D simulator
    print("Testing 3D N-body Simulator...")

    sim = NBodySimulator3D(n_particles=5)
    state = sim.initialize_random(seed=42)

    print(f"Initial state:")
    print(f"  Positions:\n{state.positions}")
    print(f"  Velocities:\n{state.velocities}")

    # Compute initial energy
    ke, pe, total = sim.compute_energy(state)
    print(f"\nInitial energy: KE={ke:.4f}, PE={pe:.4f}, Total={total:.4f}")

    # Generate trajectory
    trajectory = sim.generate_trajectory(state, n_steps=100)
    print(f"\nGenerated trajectory with {len(trajectory)} states")

    # Check energy conservation
    ke_final, pe_final, total_final = sim.compute_energy(trajectory[-1])
    print(f"Final energy: KE={ke_final:.4f}, PE={pe_final:.4f}, Total={total_final:.4f}")
    print(f"Energy drift: {abs(total_final - total):.6f}")

    # Test rotation
    rotated = rotate_state_3d(state, (45, 30, 15))
    print(f"\nRotated state (45, 30, 15 degrees):")
    print(f"  Positions:\n{rotated.positions}")

    # Generate dataset
    inputs, targets = generate_nbody_dataset_3d(n_trajectories=100, n_particles=5, seed=42)
    print(f"\nDataset shape: inputs={inputs.shape}, targets={targets.shape}")

    # Test rotation equivariance of physics
    print("\nTesting rotation equivariance of physics...")
    state1 = sim.initialize_random(seed=123)
    state1_next = sim.step(state1)

    # Apply rotation before simulation
    state1_rot = rotate_state_3d(state1, (45, 0, 0))
    state1_rot_next = sim.step(state1_rot)

    # Apply rotation after simulation
    state1_next_rot = rotate_state_3d(state1_next, (45, 0, 0))

    # Compare
    diff = np.abs(state1_rot_next.positions - state1_next_rot.positions).mean()
    print(f"  Position difference after rotation: {diff:.8f}")
    print("  (Should be near machine precision for equivariant physics)")

    print("\nSUCCESS: 3D N-body simulator working!")
