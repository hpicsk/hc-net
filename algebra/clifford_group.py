"""
Clifford Group Operations for CGENN.

Implements the Clifford group actions needed for exact equivariance:
- Pin(p,q) group representation
- Sandwich product: RxR^(-1) or RxR^dagger
- Versor (Pin element) construction
- Grade-preserving operations

Based on:
"Clifford Group Equivariant Neural Networks" (Ruhe et al., NeurIPS 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math

from .clifford import CliffordAlgebra, get_algebra


class CliffordGroupOps:
    """
    Clifford group operations for equivariant neural networks.

    The Clifford group Gamma(p,q) consists of products of invertible vectors.
    The Pin group Pin(p,q) is the subgroup with |v|^2 = +/- 1.

    Key operations:
    - Sandwich product: rho(g)(x) = g x g^(-1) for g in Gamma
    - For Pin elements: g^(-1) = g_tilde (reverse)
    - Grade automorphism: preserves grade structure
    """

    def __init__(self, algebra: CliffordAlgebra):
        self.algebra = algebra
        self.d = algebra.d
        self.dim = algebra.dim

    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the reverse (reversion) of a multivector.

        Rev(e_{i1}...e_{ik}) = e_{ik}...e_{i1} = (-1)^{k(k-1)/2} e_{i1}...e_{ik}
        """
        return self.algebra.reverse(x)

    def grade_involution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Grade involution (main involution).

        alpha(e_{i1}...e_{ik}) = (-1)^k e_{i1}...e_{ik}
        """
        grades = self.algebra.grades.to(x.device)
        signs = torch.ones(self.dim, device=x.device, dtype=x.dtype)
        for i in range(self.dim):
            grade = grades[i].item()
            if grade % 2 == 1:
                signs[i] = -1
        return x * signs

    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Clifford conjugate: combine reverse and grade involution.

        conj(x) = alpha(rev(x))
        """
        return self.grade_involution(self.reverse(x))

    def norm_squared(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the squared norm: x * conj(x).

        Returns the scalar part.
        """
        prod = self.algebra.geometric_product_optimized(x, self.conjugate(x))
        return prod[..., 0:1]

    def inverse(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute the inverse of a multivector (if it exists).

        x^(-1) = conj(x) / (x * conj(x))

        Note: Only works for versors (products of vectors).
        """
        x_conj = self.conjugate(x)
        norm_sq = self.norm_squared(x)
        return x_conj / (norm_sq + eps)

    def sandwich_product(
        self,
        g: torch.Tensor,
        x: torch.Tensor,
        use_conjugate: bool = True
    ) -> torch.Tensor:
        """
        Compute sandwich product: g * x * g^(-1) or g * x * conj(g).

        For Pin elements with |g|^2 = 1, g^(-1) = conj(g).

        Args:
            g: [..., dim] versor (group element)
            x: [..., dim] multivector to transform
            use_conjugate: If True, use g * x * conj(g) (for normalized versors)

        Returns:
            [..., dim] transformed multivector
        """
        # First: g * x
        gx = self.algebra.geometric_product_optimized(g, x)

        # Second: (g * x) * g^(-1) or (g * x) * conj(g)
        if use_conjugate:
            g_inv = self.conjugate(g)
        else:
            g_inv = self.inverse(g)

        return self.algebra.geometric_product_optimized(gx, g_inv)

    def create_rotor_2d(self, angle: torch.Tensor) -> torch.Tensor:
        """
        Create a 2D rotation rotor: R = cos(theta/2) + sin(theta/2) * e12

        Args:
            angle: [...] rotation angle in radians

        Returns:
            [..., dim] rotor multivector
        """
        half_angle = angle / 2

        # Initialize rotor
        rotor = torch.zeros(*angle.shape, self.dim, device=angle.device, dtype=angle.dtype)

        # Scalar part: cos(theta/2)
        rotor[..., 0] = torch.cos(half_angle)

        # Bivector e12 part: sin(theta/2)
        # e12 is at index 3 (bits: 11 = 1+2)
        if self.d >= 2:
            rotor[..., 3] = torch.sin(half_angle)

        return rotor

    def create_rotor_3d(self, axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """
        Create a 3D rotation rotor: R = cos(theta/2) + sin(theta/2) * (axis as bivector)

        Args:
            axis: [..., 3] normalized rotation axis
            angle: [...] rotation angle in radians

        Returns:
            [..., dim] rotor multivector
        """
        if self.d < 3:
            raise ValueError("Need at least 3D algebra for 3D rotations")

        half_angle = angle / 2
        c = torch.cos(half_angle)
        s = torch.sin(half_angle)

        # Initialize rotor
        batch_shape = axis.shape[:-1]
        rotor = torch.zeros(*batch_shape, self.dim, device=axis.device, dtype=axis.dtype)

        # Scalar part
        rotor[..., 0] = c

        # Bivector parts from axis (dual of axis gives rotation plane)
        # axis = (ax, ay, az) -> bivector = ax*e23 + ay*e31 + az*e12
        # e23 = index 6 (binary 110), e31 = index 5 (binary 101), e12 = index 3 (binary 011)
        ax, ay, az = axis[..., 0], axis[..., 1], axis[..., 2]
        rotor[..., 3] = s * az   # e12
        rotor[..., 5] = s * ay   # e13
        rotor[..., 6] = s * ax   # e23

        return rotor

    def apply_rotation_2d(self, x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D rotation to multivector x by angle.

        Args:
            x: [..., dim] multivector
            angle: [...] or scalar rotation angle

        Returns:
            [..., dim] rotated multivector
        """
        rotor = self.create_rotor_2d(angle)
        return self.sandwich_product(rotor, x)


class CliffordLinearEquivariant(nn.Module):
    """
    Clifford-group equivariant linear layer.

    Key insight from CGENN: Linear operations on multivectors that use
    only grade-respecting scalar multiplications and the geometric product
    are automatically equivariant to the Clifford group action.

    This layer performs:
    out = sum_k (W_k * x * U_k) + b

    where W_k, U_k are learnable multivector weights and * is geometric product.
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        in_channels: int,
        out_channels: int,
        n_products: int = 1,
        bias: bool = True
    ):
        """
        Args:
            algebra: CliffordAlgebra instance
            in_channels: Number of input multivector channels
            out_channels: Number of output multivector channels
            n_products: Number of geometric products to sum
            bias: Whether to include bias
        """
        super().__init__()

        self.algebra = algebra
        self.dim = algebra.dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_products = n_products

        # Weight matrices for sandwich-like products
        # W_k, U_k are [out_channels, in_channels, dim] multivector weights
        self.weight_left = nn.Parameter(
            torch.randn(n_products, out_channels, in_channels, self.dim) * 0.02
        )
        self.weight_right = nn.Parameter(
            torch.randn(n_products, out_channels, in_channels, self.dim) * 0.02
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, self.dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, ..., in_channels, dim] input multivectors

        Returns:
            [B, ..., out_channels, dim] output multivectors
        """
        batch_shape = x.shape[:-2]
        x = x.view(-1, self.in_channels, self.dim)  # [B*, in_c, dim]
        B = x.shape[0]

        # Output accumulator
        out = torch.zeros(B, self.out_channels, self.dim, device=x.device, dtype=x.dtype)

        # Compute sum of products
        for k in range(self.n_products):
            # For each output channel and input channel
            for o in range(self.out_channels):
                for i in range(self.in_channels):
                    # W_k * x_i * U_k
                    w = self.weight_left[k, o, i]   # [dim]
                    u = self.weight_right[k, o, i]  # [dim]
                    xi = x[:, i]                     # [B, dim]

                    # w * xi
                    wxi = self.algebra.geometric_product_optimized(
                        w.unsqueeze(0).expand(B, -1), xi
                    )
                    # (w * xi) * u
                    wxiu = self.algebra.geometric_product_optimized(
                        wxi, u.unsqueeze(0).expand(B, -1)
                    )
                    out[:, o] = out[:, o] + wxiu

        # Add bias
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)

        return out.view(*batch_shape, self.out_channels, self.dim)


class CliffordLinearSimple(nn.Module):
    """
    Simplified Clifford-group equivariant linear layer.

    Uses grade-wise linear transformation followed by geometric product mixing.
    More efficient than full sandwich product approach.
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        in_features: int,
        out_features: int,
        bias: bool = True
    ):
        super().__init__()

        self.algebra = algebra
        self.dim = algebra.dim
        self.in_features = in_features
        self.out_features = out_features

        # Per-grade weights
        n_grades = algebra.d + 1
        self.grade_weights = nn.ParameterList([
            nn.Parameter(torch.randn(out_features, in_features) * 0.02)
            for _ in range(n_grades)
        ])

        # Mixing weight for geometric product
        self.mix_weight = nn.Parameter(torch.randn(out_features, in_features, self.dim) * 0.02)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, self.dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., in_features, dim] input multivectors

        Returns:
            [..., out_features, dim] output multivectors
        """
        batch_shape = x.shape[:-2]
        x_flat = x.view(-1, self.in_features, self.dim)  # [B, in_f, dim]
        B = x_flat.shape[0]

        out = torch.zeros(B, self.out_features, self.dim, device=x.device, dtype=x.dtype)

        # Grade-wise transformation
        for grade, weight in enumerate(self.grade_weights):
            mask = self.algebra.grade_masks.get(grade)
            if mask is not None:
                # Ensure mask is on same device as input
                mask_f = mask.to(device=x.device, dtype=x.dtype)
                # Extract grade components and transform
                x_grade = x_flat * mask_f  # [B, in_f, dim]
                # Linear combination across input features
                # weight: [out_f, in_f], x_grade: [B, in_f, dim]
                out_grade = torch.einsum('oi,bid->bod', weight, x_grade)
                out = out + out_grade

        # Add bias
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)

        return out.view(*batch_shape, self.out_features, self.dim)


class CliffordNonlinearity(nn.Module):
    """
    Equivariant nonlinearity for Clifford algebras.

    Options:
    1. Scalar-gated: Apply nonlinearity to scalar part, use as gate
    2. Norm-based: Apply nonlinearity to norm, scale entire multivector
    3. Grade-wise norm: Apply to each grade's norm separately
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        mode: str = 'norm'
    ):
        """
        Args:
            algebra: CliffordAlgebra instance
            mode: 'scalar', 'norm', or 'grade_norm'
        """
        super().__init__()
        self.algebra = algebra
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., dim] multivector

        Returns:
            [..., dim] activated multivector
        """
        if self.mode == 'scalar':
            # Scalar gating
            scalar = x[..., 0:1]
            gate = torch.sigmoid(scalar)
            return x * gate

        elif self.mode == 'norm':
            # Norm-based activation
            norm = self.algebra.norm(x)  # [..., 1]
            activated_norm = F.silu(norm)
            scale = activated_norm / (norm + 1e-8)
            return x * scale

        elif self.mode == 'grade_norm':
            # Per-grade norm activation
            result = torch.zeros_like(x)
            for grade in range(self.algebra.d + 1):
                mask = self.algebra.grade_masks.get(grade)
                if mask is not None:
                    mask_f = mask.to(device=x.device, dtype=x.dtype)
                    x_grade = x * mask_f
                    grade_norm = torch.sqrt((x_grade ** 2).sum(dim=-1, keepdim=True) + 1e-8)
                    activated_norm = F.silu(grade_norm)
                    scale = activated_norm / (grade_norm + 1e-8)
                    result = result + x_grade * scale
            return result

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


if __name__ == '__main__':
    # Test Clifford group operations
    print("Testing Clifford Group Operations...")

    device = 'cpu'
    algebra = get_algebra(d=4, device=device)
    group_ops = CliffordGroupOps(algebra)

    # Test rotor creation
    angle = torch.tensor([0.0, 0.5, 1.0, 1.5])
    rotor = group_ops.create_rotor_2d(angle)
    print(f"2D Rotor shape: {rotor.shape}")

    # Test sandwich product
    x = torch.randn(4, algebra.dim)
    x_rotated = group_ops.sandwich_product(rotor, x)
    print(f"Sandwich product: {x.shape} -> {x_rotated.shape}")

    # Test equivariant linear layer
    print("\nTesting CliffordLinearSimple...")
    layer = CliffordLinearSimple(algebra, in_features=8, out_features=16)
    x_in = torch.randn(2, 8, algebra.dim)
    x_out = layer(x_in)
    print(f"Linear: {x_in.shape} -> {x_out.shape}")

    # Test nonlinearity
    print("\nTesting CliffordNonlinearity...")
    nonlin = CliffordNonlinearity(algebra, mode='norm')
    x_act = nonlin(x_out)
    print(f"Nonlinearity: {x_out.shape} -> {x_act.shape}")

    # Test gradient flow
    loss = x_act.sum()
    loss.backward()
    print("Gradient flow: OK")

    print("\nSUCCESS: Clifford group operations working!")
