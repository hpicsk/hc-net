"""
2D N-body Gravitational Simulation.

Generates particle trajectories for state prediction task.
Uses Velocity Verlet integration for accurate physics.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class NBodyState:
    """State of N-body system."""
    positions: np.ndarray   # [N, 2] x, y positions
    velocities: np.ndarray  # [N, 2] vx, vy velocities
    masses: np.ndarray      # [N] particle masses

    def copy(self) -> 'NBodyState':
        """Create a deep copy of the state."""
        return NBodyState(
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            masses=self.masses.copy()
        )

    def to_array(self) -> np.ndarray:
        """Convert to flat array [N, 4] with (x, y, vx, vy) per particle."""
        return np.concatenate([self.positions, self.velocities], axis=1)

    @staticmethod
    def from_array(arr: np.ndarray, masses: Optional[np.ndarray] = None) -> 'NBodyState':
        """Create from flat array [N, 4]."""
        n_particles = arr.shape[0]
        positions = arr[:, :2]
        velocities = arr[:, 2:]
        if masses is None:
            masses = np.ones(n_particles)
        return NBodyState(positions, velocities, masses)


class NBodySimulator:
    """
    2D gravitational N-body simulator using Velocity Verlet integration.

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

    def initialize_random(self, seed: Optional[int] = None) -> NBodyState:
        """
        Initialize random particle configuration.

        Particles are placed randomly in the box with small random velocities.
        """
        if seed is not None:
            np.random.seed(seed)

        # Random positions within box
        positions = np.random.uniform(
            -self.box_size / 2,
            self.box_size / 2,
            (self.n_particles, 2)
        )

        # Random velocities (relatively small)
        velocities = np.random.randn(self.n_particles, 2) * 0.5

        # Unit masses for simplicity
        masses = np.ones(self.n_particles)

        return NBodyState(positions, velocities, masses)

    def initialize_orbit(self, seed: Optional[int] = None) -> NBodyState:
        """
        Initialize particles in approximate circular orbits.

        Creates a more stable configuration for longer simulations.
        """
        if seed is not None:
            np.random.seed(seed)

        positions = np.zeros((self.n_particles, 2))
        velocities = np.zeros((self.n_particles, 2))
        masses = np.ones(self.n_particles)

        # Central heavy mass
        masses[0] = 5.0

        # Other particles in approximate orbits
        for i in range(1, self.n_particles):
            angle = 2 * np.pi * i / (self.n_particles - 1) + np.random.randn() * 0.1
            radius = 2.0 + np.random.randn() * 0.5

            positions[i] = [radius * np.cos(angle), radius * np.sin(angle)]

            # Circular orbit velocity: v = sqrt(GM/r)
            orbital_speed = np.sqrt(self.G * masses[0] / radius)
            # Perpendicular to radius
            velocities[i] = orbital_speed * np.array([-np.sin(angle), np.cos(angle)])

        return NBodyState(positions, velocities, masses)

    def compute_accelerations(self, state: NBodyState) -> np.ndarray:
        """
        Compute gravitational accelerations for all particles.

        a_i = G * sum_j m_j * (r_j - r_i) / (|r_j - r_i|^2 + softening^2)^(3/2)
        """
        N = state.positions.shape[0]
        accelerations = np.zeros((N, 2))

        for i in range(N):
            for j in range(N):
                if i != j:
                    r_ij = state.positions[j] - state.positions[i]
                    dist_sq = np.sum(r_ij ** 2) + self.softening ** 2
                    dist = np.sqrt(dist_sq)
                    accelerations[i] += self.G * state.masses[j] * r_ij / (dist * dist_sq)

        return accelerations

    def compute_accelerations_vectorized(self, state: NBodyState) -> np.ndarray:
        """Vectorized version of acceleration computation."""
        N = state.positions.shape[0]

        # Pairwise position differences: [N, N, 2]
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

    def step(self, state: NBodyState) -> NBodyState:
        """
        Advance simulation by one time step using Velocity Verlet.

        Velocity Verlet is a symplectic integrator that conserves
        energy better than simple Euler integration.
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
        temp_state = NBodyState(new_positions, state.velocities, state.masses)
        new_acc = self.compute_accelerations_vectorized(temp_state)

        # Update velocities: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        new_velocities = state.velocities + 0.5 * (acc + new_acc) * self.dt

        return NBodyState(new_positions, new_velocities, state.masses.copy())

    def generate_trajectory(
        self,
        initial_state: NBodyState,
        n_steps: int
    ) -> List[NBodyState]:
        """
        Generate trajectory of n_steps from initial state.

        Returns list of states including initial state.
        """
        trajectory = [initial_state.copy()]
        state = initial_state

        for _ in range(n_steps):
            state = self.step(state)
            trajectory.append(state.copy())

        return trajectory

    def compute_energy(self, state: NBodyState) -> Tuple[float, float, float]:
        """
        Compute total energy of the system.

        Returns (kinetic, potential, total) energy.
        """
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


def rotate_state(state: NBodyState, angle_degrees: float) -> NBodyState:
    """
    Rotate entire system by given angle.

    Used for OOD testing: train on random orientations,
    test on systematically rotated systems.

    Rotates both positions and velocities.
    """
    angle_rad = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])

    new_positions = state.positions @ rotation_matrix.T
    new_velocities = state.velocities @ rotation_matrix.T

    return NBodyState(new_positions, new_velocities, state.masses.copy())


def generate_nbody_dataset(
    n_trajectories: int = 1000,
    n_particles: int = 5,
    prediction_horizon: int = 1,
    warmup_range: Tuple[int, int] = (10, 50),
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate N-body prediction dataset.

    Each sample is a pair (input_state, target_state) where:
    - input_state: particle states at time t
    - target_state: particle states at time t + prediction_horizon

    Args:
        n_trajectories: Number of independent trajectory samples
        n_particles: Particles per system
        prediction_horizon: Steps to predict ahead
        warmup_range: Range of warmup steps to get varied configurations
        seed: Random seed

    Returns:
        inputs: [n_trajectories, n_particles, 4] (x, y, vx, vy)
        targets: [n_trajectories, n_particles, 4] state at t+horizon
    """
    np.random.seed(seed)

    simulator = NBodySimulator(n_particles=n_particles)

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


def visualize_trajectory(
    trajectory: List[NBodyState],
    output_path: Optional[str] = None
):
    """Visualize particle trajectories."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))

    n_particles = trajectory[0].positions.shape[0]
    colors = plt.cm.tab10(np.linspace(0, 1, n_particles))

    # Plot trajectories
    for i in range(n_particles):
        positions = np.array([s.positions[i] for s in trajectory])
        ax.plot(positions[:, 0], positions[:, 1], '-', color=colors[i],
                alpha=0.5, linewidth=1)
        ax.scatter(positions[0, 0], positions[0, 1], c=[colors[i]], s=100,
                   marker='o', label=f'Particle {i} (start)')
        ax.scatter(positions[-1, 0], positions[-1, 1], c=[colors[i]], s=100,
                   marker='x')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('N-body Trajectories')
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    # Test the simulator
    print("Testing N-body Simulator...")

    sim = NBodySimulator(n_particles=5)
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
    rotated = rotate_state(state, 45)
    print(f"\nRotated state (45 degrees):")
    print(f"  Positions:\n{rotated.positions}")

    # Generate dataset
    inputs, targets = generate_nbody_dataset(n_trajectories=100, n_particles=5, seed=42)
    print(f"\nDataset shape: inputs={inputs.shape}, targets={targets.shape}")

    # Visualize
    visualize_trajectory(trajectory, output_path='nbody_trajectory.png')
    print("Saved trajectory visualization to nbody_trajectory.png")
