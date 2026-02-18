"""
Mean-Field-Only 3D Classifiers (Experiments 1 & 2).

Five classifiers that demonstrate the grade hierarchy for chirality detection:
- VectorMeanField3D:  grade 0+1 only -> ~50% on chirality
- BivectorMeanField3D: + grade 2     -> ~50% on chirality, ~100% on rotation
- TrivectorMeanField3D: + grade 3    -> ~100% on both
- FullCliffordMeanField3D: all grades via Cl(3,0) -> ~100% on both
- LearnedMeanField3D: learned projection -> variable

Follows the pattern of hcnet/experiments/meanfield_only_classification.py.
"""

import torch
import torch.nn as nn
import numpy as np


class VectorMeanField3DClassifier(nn.Module):
    """
    Classifier using ONLY vector (grade-1) mean-field.

    Computes: mean(x, y, z, vx, vy, vz) over all particles -> 6D
    Then classifies from this 6D vector.

    Expected: ~50% on chirality (vectors cancel), ~50% on rotation
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 6] particle states (x, y, z, vx, vy, vz)
        Returns:
            [B, 2] class logits
        """
        mean_field = x.mean(dim=1)  # [B, 6]
        return self.classifier(mean_field)


class BivectorMeanField3DClassifier(nn.Module):
    """
    Classifier using ONLY bivector (grade-2) mean-field.

    Uses only the 3 angular momentum components (no vector features):
        L_xy = x*vy - y*vx  (angular momentum around z)
        L_xz = x*vz - z*vx  (angular momentum around y)
        L_yz = y*vz - z*vy  (angular momentum around x)

    Total: 3D mean-field = 3 bivector components only.
    No vector features are included, preventing the MLP from
    cross-correlating vector and bivector to reconstruct the trivector.

    After SO(3) rotation, the 3D angular momentum vector is rotated
    but its NORM is preserved. For both chiralities, ||L|| is the same,
    so a classifier from bivector features alone cannot distinguish them.

    Expected: ~50% on chirality, ~100% on rotation (z-axis rotation only)
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # 3 bivector only
        self.classifier = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 6] particle states
        Returns:
            [B, 2] class logits
        """
        px, py, pz = x[:, :, 0], x[:, :, 1], x[:, :, 2]
        vx, vy, vz = x[:, :, 3], x[:, :, 4], x[:, :, 5]

        # Bivector components (angular momentum per particle)
        L_xy = px * vy - py * vx  # [B, N]
        L_xz = px * vz - pz * vx  # [B, N]
        L_yz = py * vz - pz * vy  # [B, N]

        # Mean angular momentum: [B, 3]
        bivector_mean = torch.stack(
            [L_xy.mean(dim=1), L_xz.mean(dim=1), L_yz.mean(dim=1)], dim=1
        )

        return self.classifier(bivector_mean)


class TrivectorMeanField3DClassifier(nn.Module):
    """
    Classifier using ONLY the trivector (grade-3) mean-field.

    Computes the helicity pseudoscalar:
        h = mean(v_i . L_mean)  where L_mean = mean angular momentum

    This is an SO(3)-invariant pseudoscalar that changes sign under
    improper rotations (reflections). It is the key quantity that
    distinguishes left-handed from right-handed structures.

    Total: 1D mean-field = 1 trivector (helicity) only.

    Expected: ~100% on chirality, ~100% on rotation
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # 1 trivector (helicity) only
        self.classifier = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 6] particle states
        Returns:
            [B, 2] class logits
        """
        pos = x[:, :, :3]  # [B, N, 3]
        vel = x[:, :, 3:]  # [B, N, 3]

        # Angular momentum per particle: L_i = r_i x v_i  [B, N, 3]
        L = torch.cross(pos, vel, dim=-1)

        # Mean angular momentum [B, 3]
        L_mean = L.mean(dim=1)

        # Helicity (trivector/pseudoscalar): v_i . L_mean
        # SO(3)-invariant, sign-flips under improper rotation
        # For right-hand screw: v aligned with L -> positive
        # For left-hand screw: v anti-aligned with L -> negative
        L_expanded = L_mean.unsqueeze(1).expand_as(vel)  # [B, N, 3]
        helicity = (vel * L_expanded).sum(dim=2)  # [B, N]
        mean_helicity = helicity.mean(dim=1, keepdim=True)  # [B, 1]

        return self.classifier(mean_helicity)


class FullCliffordMeanField3DClassifier(nn.Module):
    """
    Classifier using full Cl(3,0) multivector mean-field.

    Uses get_algebra(3) to compute the full 8-dimensional multivector
    (1 scalar + 3 vectors + 3 bivectors + 1 pseudoscalar) from
    geometric products of embedded position, velocity, and angular
    momentum vectors.

    Key insight: The geometric product of TWO vectors only produces
    grades 0 + 2 (scalar + bivector). To get the trivector (grade 3),
    we need a TRIPLE product: pos * vel * L, where L = pos x vel.
    This produces all 8 components including the pseudoscalar.

    Expected: ~100% on chirality, ~100% on rotation
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        from nips_hcnet.algebra.clifford import get_algebra
        self.algebra = get_algebra(3, device='cpu')

        # Full Cl(3,0) has 8 components
        self.classifier = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 6] particle states
        Returns:
            [B, 2] class logits
        """
        pos = x[:, :, :3]  # [B, N, 3]
        vel = x[:, :, 3:]  # [B, N, 3]

        # Ensure algebra is on correct device
        if not hasattr(self, '_device_set') or self._device_set != x.device:
            from nips_hcnet.algebra.clifford import CliffordAlgebra
            self.algebra = CliffordAlgebra(3, device=str(x.device))
            self._device_set = x.device

        # Embed position and velocity as Cl(3,0) vectors (grade 1)
        pos_mv = self.algebra.embed_vector(pos)      # [B, N, 8]
        vel_mv = self.algebra.embed_vector(vel)      # [B, N, 8]

        # pos*vel -> grades 0+2 (scalar: pos.vel, bivector: pos^vel)
        pv_product = self.algebra.geometric_product_optimized(
            pos_mv, vel_mv
        )  # [B, N, 8]

        # Extract mean bivector (grade 2 = angular momentum)
        # In Cl(3,0): indices 4,5,6 are bivector (e12, e13, e23)
        # Mean over particles, keeping all grades
        mean_pv = pv_product.mean(dim=1, keepdim=True)  # [B, 1, 8]
        mean_pv_expanded = mean_pv.expand_as(vel_mv)     # [B, N, 8]

        # vel * mean(pos^vel): vector * (scalar+bivector) -> all grades
        # The trivector (grade 3) component captures v.L alignment (helicity)
        interaction = self.algebra.geometric_product_optimized(
            vel_mv, mean_pv_expanded
        )  # [B, N, 8]

        # Mean-field: average multivector over particles
        mean_mv = interaction.mean(dim=1)  # [B, 8]

        return self.classifier(mean_mv)


class LearnedMeanField3DClassifier(nn.Module):
    """
    Classifier with learned per-particle projection before averaging.

    Projects each particle to a 16D feature space, then averages.
    Tests whether learned representations can capture trivector-like features.

    Expected: variable (may partially learn trivector features)
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        meanfield_dim: int = 16,
    ):
        super().__init__()

        self.particle_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, meanfield_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(meanfield_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 6] particle states
        Returns:
            [B, 2] class logits
        """
        h = self.particle_proj(x)  # [B, N, meanfield_dim]
        mean_field = h.mean(dim=1)  # [B, meanfield_dim]
        return self.classifier(mean_field)


if __name__ == '__main__':
    print("Testing Mean-Field 3D Classifiers...")

    B, N = 4, 10
    x = torch.randn(B, N, 6)

    models = {
        'VectorMeanField3D': VectorMeanField3DClassifier(),
        'BivectorMeanField3D': BivectorMeanField3DClassifier(),
        'TrivectorMeanField3D': TrivectorMeanField3DClassifier(),
        'FullCliffordMeanField3D': FullCliffordMeanField3DClassifier(),
        'LearnedMeanField3D': LearnedMeanField3DClassifier(),
    }

    for name, model in models.items():
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        model.zero_grad()
        print(f"  {name}: output={logits.shape}, grad_flow=OK")

    print("\nSUCCESS: All mean-field 3D classifiers working!")
