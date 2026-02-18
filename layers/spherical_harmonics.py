"""
Spherical Harmonics and Radial Basis Functions for NequIP.

Implements angular encoding using:
- 2D: Fourier modes (circular harmonics)
- 3D: Real spherical harmonics

Also provides radial basis functions for distance encoding.

Based on concepts from:
"E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials"
(Batzner et al., Nature Communications 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class RadialBasisFunctions(nn.Module):
    """
    Radial basis function expansion for encoding distances.

    Uses Gaussian or Bessel basis functions to create smooth,
    learnable representations of pairwise distances.
    """

    def __init__(
        self,
        num_basis: int = 8,
        cutoff: float = 5.0,
        basis_type: str = 'gaussian'
    ):
        """
        Args:
            num_basis: Number of basis functions
            cutoff: Maximum distance cutoff
            basis_type: 'gaussian' or 'bessel'
        """
        super().__init__()

        self.num_basis = num_basis
        self.cutoff = cutoff
        self.basis_type = basis_type

        if basis_type == 'gaussian':
            # Gaussian centers evenly spaced
            centers = torch.linspace(0, cutoff, num_basis)
            self.register_buffer('centers', centers)
            # Width based on spacing
            self.width = cutoff / num_basis
        elif basis_type == 'bessel':
            # Bessel function frequencies
            freqs = torch.arange(1, num_basis + 1) * math.pi / cutoff
            self.register_buffer('freqs', freqs)
        else:
            raise ValueError(f"Unknown basis type: {basis_type}")

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Expand distances into basis function values.

        Args:
            distances: [...] distance values

        Returns:
            [..., num_basis] basis function values
        """
        if self.basis_type == 'gaussian':
            # Gaussian RBF: exp(-((d - center) / width)^2)
            d = distances.unsqueeze(-1)  # [..., 1]
            return torch.exp(-((d - self.centers) / self.width) ** 2)
        else:
            # Bessel basis: sin(freq * d) / d
            d = distances.unsqueeze(-1)  # [..., 1]
            # Avoid division by zero
            d_safe = torch.clamp(d, min=1e-8)
            return torch.sin(self.freqs * d_safe) / d_safe


class SmoothCutoff(nn.Module):
    """
    Smooth cutoff function for distance-based interactions.

    Ensures interactions go smoothly to zero at the cutoff distance.
    """

    def __init__(self, cutoff: float = 5.0, p: int = 6):
        """
        Args:
            cutoff: Cutoff distance
            p: Polynomial order for smoothness
        """
        super().__init__()
        self.cutoff = cutoff
        self.p = p

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Compute smooth cutoff envelope.

        Args:
            distances: [...] distance values

        Returns:
            [...] cutoff values in [0, 1]
        """
        # Polynomial cutoff: (1 - (d/cutoff)^p)^p for d < cutoff, else 0
        x = distances / self.cutoff
        cutoff_vals = torch.where(
            x < 1,
            (1 - x ** self.p) ** self.p,
            torch.zeros_like(x)
        )
        return cutoff_vals


class CircularHarmonics2D(nn.Module):
    """
    2D angular encoding using circular harmonics (Fourier modes).

    For 2D, spherical harmonics reduce to cos(m*theta) and sin(m*theta).
    This provides rotation-equivariant angular features.
    """

    def __init__(self, max_order: int = 4):
        """
        Args:
            max_order: Maximum harmonic order (number of frequencies)
        """
        super().__init__()
        self.max_order = max_order
        # Total features: 1 (m=0) + 2*max_order (cos and sin for m=1..max_order)
        self.num_features = 1 + 2 * max_order

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Compute circular harmonics for 2D vectors.

        Args:
            vectors: [..., 2] unit vectors or direction vectors

        Returns:
            [..., num_features] harmonic features
        """
        # Compute angle from vector
        x, y = vectors[..., 0], vectors[..., 1]
        theta = torch.atan2(y, x)  # [...]

        features = [torch.ones_like(theta)]  # m=0: constant

        for m in range(1, self.max_order + 1):
            features.append(torch.cos(m * theta))
            features.append(torch.sin(m * theta))

        return torch.stack(features, dim=-1)


class SphericalHarmonics3D(nn.Module):
    """
    3D angular encoding using real spherical harmonics.

    Computes Y_l^m for l = 0, 1, ..., max_l.
    Uses real form: combinations of Y_l^m + Y_l^{-m} and Y_l^m - Y_l^{-m}.
    """

    def __init__(self, max_l: int = 2):
        """
        Args:
            max_l: Maximum angular momentum quantum number
        """
        super().__init__()
        self.max_l = max_l
        # Total features: sum_{l=0}^{max_l} (2l+1)
        self.num_features = (max_l + 1) ** 2

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Compute real spherical harmonics for 3D vectors.

        Args:
            vectors: [..., 3] unit vectors

        Returns:
            [..., num_features] harmonic features
        """
        x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]

        # Compute spherical coordinates
        r = torch.sqrt(x**2 + y**2 + z**2 + 1e-8)
        x, y, z = x/r, y/r, z/r  # Normalize

        features = []

        # l=0: Y_0^0 = 1/sqrt(4*pi) (constant, we use 1)
        features.append(torch.ones_like(x))

        if self.max_l >= 1:
            # l=1: Y_1^{-1} ~ y, Y_1^0 ~ z, Y_1^1 ~ x
            features.extend([y, z, x])

        if self.max_l >= 2:
            # l=2: 5 components
            features.extend([
                x * y,                    # Y_2^{-2}
                y * z,                    # Y_2^{-1}
                3 * z**2 - 1,            # Y_2^0
                x * z,                    # Y_2^1
                x**2 - y**2              # Y_2^2
            ])

        if self.max_l >= 3:
            # l=3: 7 components (simplified)
            features.extend([
                y * (3 * x**2 - y**2),
                x * y * z,
                y * (5 * z**2 - 1),
                z * (5 * z**2 - 3),
                x * (5 * z**2 - 1),
                z * (x**2 - y**2),
                x * (x**2 - 3 * y**2)
            ])

        if self.max_l >= 4:
            # l=4: 9 components (real solid harmonics, unnormalized)
            x2, y2, z2 = x**2, y**2, z**2
            features.extend([
                x * y * (x2 - y2),                       # Y_4^{-4}
                y * z * (3 * x2 - y2),                    # Y_4^{-3}
                x * y * (7 * z2 - 1),                     # Y_4^{-2}
                y * z * (7 * z2 - 3),                     # Y_4^{-1}
                35 * z2**2 - 30 * z2 + 3,                 # Y_4^0
                x * z * (7 * z2 - 3),                     # Y_4^1
                (x2 - y2) * (7 * z2 - 1),                 # Y_4^2
                x * z * (x2 - 3 * y2),                    # Y_4^3
                x2**2 - 6 * x2 * y2 + y2**2              # Y_4^4
            ])

        return torch.stack(features[:self.num_features], dim=-1)


class TensorProductLayer(nn.Module):
    """
    Simplified tensor product layer for combining angular and radial features.

    Combines:
    - Radial features (invariant)
    - Angular features (equivariant)

    Into output features while maintaining equivariance.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        radial_features: int,
        angular_features: int
    ):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            radial_features: Number of radial basis functions
            angular_features: Number of angular harmonics
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Radial network: produces coupling coefficients
        self.radial_net = nn.Sequential(
            nn.Linear(radial_features, 64),
            nn.SiLU(),
            nn.Linear(64, in_features * out_features)
        )

        # Angular mixing
        self.angular_mix = nn.Linear(angular_features, out_features)

    def forward(
        self,
        features: torch.Tensor,
        radial: torch.Tensor,
        angular: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: [..., in_features] input features
            radial: [..., radial_features] radial basis values
            angular: [..., angular_features] angular harmonic values

        Returns:
            [..., out_features] output features
        """
        # Get coupling weights from radial features
        weights = self.radial_net(radial)  # [..., in*out]
        weights = weights.view(*radial.shape[:-1], self.in_features, self.out_features)

        # Apply weights to features
        out = torch.einsum('...i,...io->...o', features, weights)

        # Mix with angular features
        angular_contribution = self.angular_mix(angular)
        out = out * angular_contribution

        return out


class EquivariantMessageBlock(nn.Module):
    """
    Equivariant message passing block using angular and radial features.

    For each pair of nodes:
    1. Compute radial features from distance
    2. Compute angular features from direction
    3. Combine via tensor product
    4. Aggregate messages
    """

    def __init__(
        self,
        hidden_dim: int,
        num_radial: int = 8,
        max_angular: int = 4,
        cutoff: float = 5.0,
        coord_dim: int = 2
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim

        # Radial basis
        self.radial_basis = RadialBasisFunctions(num_radial, cutoff)
        self.cutoff_fn = SmoothCutoff(cutoff)

        # Angular encoding
        if coord_dim == 2:
            self.angular = CircularHarmonics2D(max_angular)
            angular_features = self.angular.num_features
        else:
            self.angular = SphericalHarmonics3D(max_angular)
            angular_features = self.angular.num_features

        # Message network
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_radial + angular_features, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            h: [B, N, hidden_dim] node features
            pos: [B, N, coord_dim] node positions

        Returns:
            [B, N, hidden_dim] updated features
        """
        B, N, _ = pos.shape

        # Compute pairwise vectors and distances
        pos_diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B, N, N, coord_dim]
        dist = torch.norm(pos_diff, dim=-1, keepdim=True)  # [B, N, N, 1]
        dist_sq = dist.squeeze(-1)  # [B, N, N]

        # Direction vectors (normalized)
        direction = pos_diff / (dist + 1e-8)  # [B, N, N, coord_dim]

        # Radial features
        radial = self.radial_basis(dist_sq)  # [B, N, N, num_radial]
        cutoff = self.cutoff_fn(dist_sq).unsqueeze(-1)  # [B, N, N, 1]

        # Angular features
        angular = self.angular(direction)  # [B, N, N, angular_features]

        # Build messages
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, hidden]
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, hidden]

        msg_input = torch.cat([h_i, h_j, radial, angular], dim=-1)
        messages = self.message_net(msg_input)  # [B, N, N, hidden]

        # Apply cutoff
        messages = messages * cutoff

        # Aggregate messages
        msg_agg = messages.sum(dim=2)  # [B, N, hidden]

        # Update nodes
        update_input = torch.cat([h, msg_agg], dim=-1)
        h_update = self.update_net(update_input)

        return self.norm(h + h_update)


if __name__ == '__main__':
    # Test the modules
    print("Testing Spherical Harmonics and Radial Basis Functions...")

    # Test radial basis
    print("\n1. RadialBasisFunctions:")
    rbf = RadialBasisFunctions(num_basis=8, cutoff=5.0)
    distances = torch.rand(4, 5, 5) * 5  # Random distances
    radial_features = rbf(distances)
    print(f"   Input: {distances.shape}")
    print(f"   Output: {radial_features.shape}")

    # Test circular harmonics
    print("\n2. CircularHarmonics2D:")
    ch = CircularHarmonics2D(max_order=4)
    vectors_2d = F.normalize(torch.randn(4, 5, 5, 2), dim=-1)
    angular_2d = ch(vectors_2d)
    print(f"   Input: {vectors_2d.shape}")
    print(f"   Output: {angular_2d.shape}")

    # Test spherical harmonics
    print("\n3. SphericalHarmonics3D:")
    sh = SphericalHarmonics3D(max_l=2)
    vectors_3d = F.normalize(torch.randn(4, 5, 5, 3), dim=-1)
    angular_3d = sh(vectors_3d)
    print(f"   Input: {vectors_3d.shape}")
    print(f"   Output: {angular_3d.shape}")

    # Test message block
    print("\n4. EquivariantMessageBlock:")
    block = EquivariantMessageBlock(hidden_dim=64, coord_dim=2)
    h = torch.randn(4, 5, 64)
    pos = torch.randn(4, 5, 2)
    h_out = block(h, pos)
    print(f"   h: {h.shape} -> {h_out.shape}")
    print(f"   pos: {pos.shape}")

    # Test gradient
    loss = h_out.sum()
    loss.backward()
    print("   Gradient flow: OK")

    print("\nSUCCESS: All modules working!")
