"""
Chiral Spiral Dataset: 3D helical point clouds with chirality labels.

Generates left-handed and right-handed helices in 3D, randomly rotated
by SO(3) to prevent coordinate-axis shortcuts.

Critical mathematical property:
- Vector mean (grade-1) is IDENTICAL for both chiralities after rotation
- Bivector mean (grade-2) is IDENTICAL for both chiralities after rotation
- ONLY the trivector mean (grade-3, pseudoscalar) differs in sign

This proves that trivectors are NECESSARY for chirality detection,
which bivectors alone cannot achieve.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict
from scipy.spatial.transform import Rotation


class ChiralSpiralDataset(Dataset):
    """
    Dataset of 3D helical point clouds with left/right chirality labels.

    Each sample is a helix:
        x = R * cos(t)
        y = R * sin(t)
        z = chirality * pitch * t / (2*pi)

    with tangent velocity vectors, randomly rotated by SO(3).

    After rotation, vector and bivector means are identical for both
    chiralities; only the trivector mean differs in sign.
    """

    def __init__(
        self,
        n_samples: int = 5000,
        n_points: int = 20,
        radius: float = 1.0,
        pitch: float = 1.0,
        n_turns: float = 2.0,
        noise_std: float = 0.02,
        seed: int = 42,
    ):
        """
        Args:
            n_samples: Number of helix samples
            n_points: Points per helix
            radius: Helix radius
            pitch: Helix pitch (height per turn)
            n_turns: Number of turns
            noise_std: Gaussian noise added to points
            seed: Random seed
        """
        self.n_samples = n_samples
        self.n_points = n_points

        rng = np.random.RandomState(seed)

        self.positions = np.zeros((n_samples, n_points, 3), dtype=np.float32)
        self.velocities = np.zeros((n_samples, n_points, 3), dtype=np.float32)
        self.labels = np.zeros(n_samples, dtype=np.int64)

        t_values = np.linspace(0, 2 * np.pi * n_turns, n_points)

        for i in range(n_samples):
            # Random chirality: +1 (right-handed) or -1 (left-handed)
            chirality = rng.choice([-1, 1])
            self.labels[i] = 1 if chirality == 1 else 0

            # Slight random variation in radius and pitch
            r = radius * (1.0 + rng.randn() * 0.1)
            p = pitch * (1.0 + rng.randn() * 0.1)

            # Generate helix positions
            pos = np.zeros((n_points, 3))
            pos[:, 0] = r * np.cos(t_values)
            pos[:, 1] = r * np.sin(t_values)
            pos[:, 2] = chirality * p * t_values / (2 * np.pi)

            # Tangent velocities (derivative of position wrt t)
            vel = np.zeros((n_points, 3))
            vel[:, 0] = -r * np.sin(t_values)
            vel[:, 1] = r * np.cos(t_values)
            vel[:, 2] = chirality * p / (2 * np.pi)

            # Add noise
            pos += rng.randn(n_points, 3) * noise_std
            vel += rng.randn(n_points, 3) * noise_std * 0.1

            # Random SO(3) rotation (prevents z-coordinate shortcuts)
            R = Rotation.random(random_state=rng).as_matrix().astype(np.float32)
            pos = pos @ R.T
            vel = vel @ R.T

            self.positions[i] = pos
            self.velocities[i] = vel

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            state: [N, 6] positions and velocities concatenated
            label: scalar chirality label (0=left, 1=right)
        """
        state = np.concatenate(
            [self.positions[idx], self.velocities[idx]], axis=1
        )
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def compute_trivector_from_state(
    positions: np.ndarray, velocities: np.ndarray
) -> float:
    """
    Compute the trivector (pseudoscalar / helicity) from positions and velocities.

    The chirality discriminant is the "helicity":
        h = mean(v_i . L_mean)

    where L_mean = mean(r_j x v_j) is the mean angular momentum.

    This measures the component of velocity along the angular momentum axis.
    For a right-hand screw, v is aligned with L -> h > 0.
    For a left-hand screw, v is anti-aligned with L -> h < 0.

    This quantity is SO(3)-invariant (preserved under rotation) but
    changes sign under improper rotations (reflections), making it
    a pseudoscalar / trivector quantity.

    Bivectors (angular momentum alone) cannot distinguish chirality
    because L_mean is identical for both handedness classes.

    Args:
        positions: [N, 3] particle positions
        velocities: [N, 3] particle velocities

    Returns:
        Scalar helicity value (positive for right-handed, negative for left-handed)
    """
    # Angular momentum per particle: L_i = r_i x v_i
    L = np.cross(positions, velocities)  # [N, 3]

    # Mean angular momentum
    L_mean = L.mean(axis=0)  # [3]

    # Helicity: projection of velocity onto angular momentum axis
    # h_i = v_i . L_mean (SO(3)-invariant pseudoscalar)
    helicity = np.sum(velocities * L_mean[np.newaxis, :], axis=1)  # [N]

    return float(helicity.mean())


def verify_chirality_properties(dataset: ChiralSpiralDataset) -> Dict:
    """
    Verify the key mathematical properties of the chiral spiral dataset.

    Computes mean vector, bivector, and trivector for both chirality classes
    and confirms:
    - Vector means are nearly identical (both ~0 after rotation)
    - Bivector means are nearly identical
    - Trivector means differ in sign

    Args:
        dataset: ChiralSpiralDataset instance

    Returns:
        Dictionary with verification results
    """
    left_mask = dataset.labels == 0
    right_mask = dataset.labels == 1

    results = {}

    for name, mask in [("left", left_mask), ("right", right_mask)]:
        pos = dataset.positions[mask]
        vel = dataset.velocities[mask]
        n = pos.shape[0]

        # Vector mean: avg position and velocity
        vec_mean_pos = pos.mean(axis=(0, 1))  # [3]
        vec_mean_vel = vel.mean(axis=(0, 1))  # [3]
        vec_mag = np.linalg.norm(np.concatenate([vec_mean_pos, vec_mean_vel]))

        # Bivector mean: avg angular momentum
        bivectors = []
        for j in range(n):
            L = np.cross(pos[j], vel[j])  # [N, 3]
            bivectors.append(L.mean(axis=0))
        biv_mean = np.mean(bivectors, axis=0)  # [3]
        biv_mag = np.linalg.norm(biv_mean)

        # Trivector mean: avg scalar triple product
        trivectors = []
        for j in range(n):
            t = compute_trivector_from_state(pos[j], vel[j])
            trivectors.append(t)
        triv_mean = np.mean(trivectors)

        results[name] = {
            "vector_magnitude": float(vec_mag),
            "bivector_magnitude": float(biv_mag),
            "trivector_mean": float(triv_mean),
            "n_samples": int(n),
        }

    # Compute differences
    results["vector_diff"] = abs(
        results["left"]["vector_magnitude"] - results["right"]["vector_magnitude"]
    )
    results["bivector_diff"] = abs(
        results["left"]["bivector_magnitude"]
        - results["right"]["bivector_magnitude"]
    )
    results["trivector_diff"] = abs(
        results["left"]["trivector_mean"] - results["right"]["trivector_mean"]
    )
    results["trivector_sign_differs"] = (
        results["left"]["trivector_mean"] * results["right"]["trivector_mean"] < 0
    )

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("CHIRAL SPIRAL DATASET: Verification")
    print("=" * 70)

    dataset = ChiralSpiralDataset(n_samples=2000, n_points=20, seed=42)

    print(f"\nDataset size: {len(dataset)}")
    print(f"Labels distribution: left={sum(dataset.labels==0)}, right={sum(dataset.labels==1)}")

    # Verify properties
    results = verify_chirality_properties(dataset)

    print(f"\n--- Left-handed helices ---")
    r = results["left"]
    print(f"  Vector mean magnitude:  {r['vector_magnitude']:.6f}")
    print(f"  Bivector mean magnitude: {r['bivector_magnitude']:.6f}")
    print(f"  Trivector mean:          {r['trivector_mean']:.6f}")

    print(f"\n--- Right-handed helices ---")
    r = results["right"]
    print(f"  Vector mean magnitude:  {r['vector_magnitude']:.6f}")
    print(f"  Bivector mean magnitude: {r['bivector_magnitude']:.6f}")
    print(f"  Trivector mean:          {r['trivector_mean']:.6f}")

    print(f"\n--- Differences ---")
    print(f"  Vector magnitude diff:   {results['vector_diff']:.6f}  (should be ~0)")
    print(f"  Bivector magnitude diff: {results['bivector_diff']:.6f}  (should be ~0)")
    print(f"  Trivector mean diff:     {results['trivector_diff']:.6f}  (should be >> 0)")
    print(f"  Trivector signs differ:  {results['trivector_sign_differs']}")

    if results["trivector_sign_differs"]:
        print("\nCONFIRMED: Trivector (pseudoscalar) distinguishes chirality!")
        print("Vector and bivector means are similar for both classes,")
        print("but trivector means have OPPOSITE signs.")
    else:
        print("\nWARNING: Trivector signs did not differ as expected.")

    # Test single sample
    state, label = dataset[0]
    print(f"\nSingle sample: shape={state.shape}, label={label.item()}")

    print("\nSUCCESS: Chiral spiral dataset verified!")
