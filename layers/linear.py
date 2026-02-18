"""
Clifford Linear Layer.

Implements linear transformation in Clifford algebra space
using geometric product for weight-input interaction.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import math
from typing import Optional

from ..algebra.clifford import CliffordAlgebra, get_algebra


class CliffordLinear(nn.Module):
    """
    Linear layer operating on multivector representations.

    Unlike standard linear layers that use scalar multiplication,
    this layer uses the geometric product, allowing weights to
    encode rotations and reflections.

    Args:
        in_features: Number of input features (each is a multivector)
        out_features: Number of output features
        algebra_dim: Dimension of the Clifford algebra generators (d)
        bias: Whether to include bias
        mode: 'full' (use all grades) or 'even' (use only even grades)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        algebra_dim: int = 8,
        bias: bool = True,
        mode: str = 'full',
        device: str = 'cuda'
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.algebra_dim = algebra_dim
        self.mv_dim = 2 ** algebra_dim
        self.mode = mode
        self.device_str = device

        # Get algebra instance
        self.algebra = get_algebra(algebra_dim, device)

        # Precompute dense Cayley table for fast computation
        if not hasattr(self.algebra, '_cayley_dense'):
            self.algebra.geometric_product_optimized(
                torch.zeros(1, self.mv_dim, device=device),
                torch.zeros(1, self.mv_dim, device=device)
            )
        self.register_buffer('cayley', self.algebra._cayley_dense)

        # Weight: [out_features, in_features, mv_dim]
        # Each weight is a multivector
        self.weight = nn.Parameter(torch.empty(out_features, in_features, self.mv_dim))

        if bias:
            # Bias is also a multivector per output feature
            self.bias = nn.Parameter(torch.empty(out_features, self.mv_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using adapted Kaiming initialization."""
        # Initialize primarily in the scalar and vector components
        # to start close to standard linear behavior

        # Scalar part: standard initialization
        fan_in = self.in_features * self.mv_dim
        std = 1.0 / math.sqrt(fan_in)

        # Initialize scalar component (index 0)
        init.normal_(self.weight[:, :, 0], std=std)

        # Initialize vector components (indices 2^i) with smaller values
        for i in range(self.algebra_dim):
            init.normal_(self.weight[:, :, 2**i], std=std * 0.5)

        # Initialize higher grades with even smaller values
        for idx in range(self.mv_dim):
            grade = bin(idx).count('1')
            if grade > 1:
                init.normal_(self.weight[:, :, idx], std=std * 0.1)

        if self.bias is not None:
            # Bias starts as zero multivector
            init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, in_features, mv_dim] input multivectors
               OR [B, in_features * mv_dim] flattened

        Returns:
            [B, out_features, mv_dim] output multivectors
        """
        # Handle flattened input
        if x.dim() == 2:
            B = x.shape[0]
            x = x.view(B, self.in_features, self.mv_dim)
        else:
            B = x.shape[0]

        # Compute geometric product between inputs and weights
        # x: [B, in_f, mv_dim]
        # weight: [out_f, in_f, mv_dim]

        # Use einsum for batched geometric product
        # For each (b, o, i): compute geom_prod(x[b,i], weight[o,i])
        # Then sum over i

        # Expand dimensions for broadcasting
        # x: [B, 1, in_f, mv_dim, 1]
        # weight: [1, out_f, in_f, 1, mv_dim]
        # cayley: [mv_dim, mv_dim, mv_dim]

        # Geometric product via einsum
        # products[b, o, i, k] = sum_{a,b} x[b,i,a] * weight[o,i,b] * cayley[a,b,k]
        products = torch.einsum('bia,oib,abk->boik', x, self.weight, self.cayley)

        # Sum over input features
        output = products.sum(dim=2)  # [B, out_f, mv_dim]

        if self.bias is not None:
            output = output + self.bias.unsqueeze(0)

        return output

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'algebra_dim={self.algebra_dim}, bias={self.bias is not None}')


class CliffordLinearSimple(nn.Module):
    """
    Simplified Clifford linear layer using separate scalar and geometric operations.

    This layer decomposes the operation into:
    1. Standard linear transformation (scalar part)
    2. Geometric interaction (bivector generation)

    More efficient for larger dimensions.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        algebra_dim: int = 8,
        bias: bool = True,
        device: str = 'cuda'
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.algebra_dim = algebra_dim
        self.mv_dim = 2 ** algebra_dim

        # Standard linear for scalar/vector parts
        self.linear = nn.Linear(in_features * (1 + algebra_dim),
                               out_features * (1 + algebra_dim),
                               bias=bias)

        # Bivector generation: learn pairwise interactions
        # Number of bivectors = C(algebra_dim, 2)
        self.num_bivectors = algebra_dim * (algebra_dim - 1) // 2

        if self.num_bivectors > 0:
            self.bivector_weight = nn.Parameter(
                torch.empty(out_features, in_features, self.num_bivectors)
            )
            init.xavier_uniform_(self.bivector_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with separated scalar/vector and bivector computation.

        Args:
            x: [B, in_features, mv_dim] or [B, in_features, 1 + d] (scalar + vector)

        Returns:
            [B, out_features, mv_dim] output multivector
        """
        B = x.shape[0]

        # Extract scalar and vector parts
        if x.shape[-1] == self.mv_dim:
            # Full multivector input
            scalar = x[:, :, 0:1]  # [B, in_f, 1]
            vectors = []
            for i in range(self.algebra_dim):
                vectors.append(x[:, :, 2**i:2**i+1])
            vector = torch.cat(vectors, dim=-1)  # [B, in_f, d]
            scalar_vector = torch.cat([scalar, vector], dim=-1)  # [B, in_f, 1+d]
        else:
            scalar_vector = x  # Assume already [B, in_f, 1+d]

        # Flatten for linear layer
        sv_flat = scalar_vector.view(B, -1)  # [B, in_f * (1+d)]

        # Standard linear transformation
        sv_out = self.linear(sv_flat)  # [B, out_f * (1+d)]
        sv_out = sv_out.view(B, self.out_features, 1 + self.algebra_dim)

        # Build output multivector
        output = torch.zeros(B, self.out_features, self.mv_dim,
                           device=x.device, dtype=x.dtype)

        # Set scalar
        output[:, :, 0] = sv_out[:, :, 0]

        # Set vectors
        for i in range(self.algebra_dim):
            output[:, :, 2**i] = sv_out[:, :, 1 + i]

        # Generate bivectors from input vectors
        if self.num_bivectors > 0:
            # For each output, compute weighted sum of input bivector generators
            # v_i ∧ v_j = v_i * v_j - v_j * v_i (antisymmetric)

            # Extract input vectors
            in_vectors = sv_out.view(B, self.out_features, 1 + self.algebra_dim)[:, :, 1:]

            # Compute pairwise products (simplified as outer products)
            biv_idx = 0
            for i in range(self.algebra_dim):
                for j in range(i + 1, self.algebra_dim):
                    # Bivector index in multivector is 2^i + 2^j
                    mv_idx = 2**i + 2**j
                    # Simple approximation: v_i * v_j - v_j * v_i ~ 2 * v_i * v_j for small values
                    output[:, :, mv_idx] = in_vectors[:, :, i] * in_vectors[:, :, j]
                    biv_idx += 1

        return output


class ProjectionHead(nn.Module):
    """
    Projection head for extracting scalar logits from multivector.

    Used as the final classification layer.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        algebra_dim: int = 8,
        mode: str = 'scalar'
    ):
        super().__init__()

        self.in_features = in_features
        self.num_classes = num_classes
        self.algebra_dim = algebra_dim
        self.mv_dim = 2 ** algebra_dim
        self.mode = mode

        if mode == 'scalar':
            # Only use scalar components
            self.fc = nn.Linear(in_features, num_classes)
        elif mode == 'scalar_vector':
            # Use scalar + vector components
            self.fc = nn.Linear(in_features * (1 + algebra_dim), num_classes)
        elif mode == 'all':
            # Use all multivector components
            self.fc = nn.Linear(in_features * self.mv_dim, num_classes)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project multivector to class logits.

        Args:
            x: [B, in_features, mv_dim] multivector features

        Returns:
            [B, num_classes] logits
        """
        B = x.shape[0]

        if self.mode == 'scalar':
            features = x[:, :, 0]  # [B, in_features]
        elif self.mode == 'scalar_vector':
            scalar = x[:, :, 0:1]
            vectors = []
            for i in range(self.algebra_dim):
                vectors.append(x[:, :, 2**i:2**i+1])
            features = torch.cat([scalar] + vectors, dim=-1)
            features = features.view(B, -1)
        else:
            features = x.view(B, -1)

        return self.fc(features)
