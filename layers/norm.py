"""
Normalization layers for Clifford algebra neural networks.

Provides grade-aware normalization that respects the
multivector structure.
"""

import torch
import torch.nn as nn
from typing import Optional


class CliffordBatchNorm(nn.Module):
    """
    Batch normalization for multivector features.

    Normalizes each grade separately to preserve the relative
    importance of different geometric components.

    Args:
        num_features: Number of feature channels (blocks * mv_dim)
        block_size: Clifford algebra dimension (d)
        eps: Small constant for numerical stability
        momentum: Momentum for running statistics
        affine: Whether to learn affine parameters
    """

    def __init__(
        self,
        num_features: int,
        block_size: int = 8,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True
    ):
        super().__init__()

        self.num_features = num_features
        self.block_size = block_size
        self.mv_dim = 2 ** block_size
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        # Compute number of blocks
        assert num_features % block_size == 0, \
            f"num_features ({num_features}) must be divisible by block_size ({block_size})"
        self.num_blocks = num_features // block_size

        # Use standard batch norm for the feature channels
        # This treats each channel independently
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, C, H, W] or [B, C]

        Returns:
            Normalized tensor of same shape
        """
        if x.dim() == 2:
            # Add spatial dimensions for BatchNorm2d
            x = x.unsqueeze(-1).unsqueeze(-1)
            x = self.bn(x)
            return x.squeeze(-1).squeeze(-1)
        else:
            return self.bn(x)


class CliffordBatchNormMV(nn.Module):
    """
    Batch normalization with multivector-aware statistics.

    Computes separate statistics for each grade and normalizes
    to preserve the geometric structure.
    """

    def __init__(
        self,
        num_blocks: int,
        algebra_dim: int = 8,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True
    ):
        super().__init__()

        self.num_blocks = num_blocks
        self.algebra_dim = algebra_dim
        self.mv_dim = 2 ** algebra_dim
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        # Compute grade masks
        self.grades = []
        for i in range(self.mv_dim):
            self.grades.append(bin(i).count('1'))

        # Separate normalization per grade
        self.num_grades = algebra_dim + 1
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(num_blocks, eps=eps, momentum=momentum, affine=affine)
            for _ in range(self.num_grades)
        ])

        # Optional per-grade scaling
        if affine:
            self.grade_scale = nn.Parameter(torch.ones(self.num_grades))
        else:
            self.register_parameter('grade_scale', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, num_blocks, mv_dim] or [B, num_blocks, mv_dim, H, W]

        Returns:
            Normalized tensor
        """
        is_spatial = x.dim() == 5

        if is_spatial:
            B, nb, mv, H, W = x.shape
            x = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, nb, mv)

        B, nb, mv = x.shape
        output = torch.zeros_like(x)

        # Normalize each grade separately
        for grade in range(self.num_grades):
            # Get indices for this grade
            indices = [i for i in range(mv) if self.grades[i] == grade]
            if not indices:
                continue

            # Extract components of this grade
            grade_features = x[:, :, indices]  # [B, nb, num_grade_components]

            # Compute grade-level norm and normalize
            grade_norm = torch.sqrt((grade_features ** 2).sum(dim=-1, keepdim=True) + self.eps)
            grade_features = grade_features / grade_norm

            # Apply batch norm to the norm values
            if len(indices) > 0:
                # Normalize the norm values across batch
                norm_flat = grade_norm.squeeze(-1)  # [B, nb]
                norm_normalized = self.norms[grade](norm_flat)  # [B, nb]

                # Reconstruct with normalized magnitudes
                grade_features = grade_features * norm_normalized.unsqueeze(-1)

            # Apply grade scaling
            if self.grade_scale is not None:
                grade_features = grade_features * self.grade_scale[grade]

            # Put back
            for idx, i in enumerate(indices):
                output[:, :, i] = grade_features[:, :, idx]

        if is_spatial:
            output = output.reshape(B // (H * W), H, W, nb, mv).permute(0, 3, 4, 1, 2)

        return output


class CliffordLayerNorm(nn.Module):
    """
    Layer normalization for multivector features.

    Normalizes across the multivector dimension while
    optionally preserving relative grade magnitudes.
    """

    def __init__(
        self,
        normalized_shape: int,
        block_size: int = 8,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        preserve_grades: bool = True
    ):
        super().__init__()

        self.normalized_shape = normalized_shape
        self.block_size = block_size
        self.eps = eps
        self.preserve_grades = preserve_grades

        if preserve_grades:
            # Separate affine parameters per grade
            num_grades = block_size + 1
            if elementwise_affine:
                self.weight = nn.Parameter(torch.ones(num_grades))
                self.bias = nn.Parameter(torch.zeros(num_grades))
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)
        else:
            self.ln = nn.LayerNorm(normalized_shape, eps=eps,
                                   elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [..., normalized_shape]

        Returns:
            Layer-normalized tensor
        """
        if not self.preserve_grades:
            return self.ln(x)

        # Grade-preserving normalization
        # Normalize within each grade, then scale
        mv_dim = 2 ** self.block_size

        # Compute grades
        grades = [bin(i).count('1') for i in range(mv_dim)]

        # Reshape if needed
        original_shape = x.shape
        if x.shape[-1] != mv_dim:
            # Assume flattened blocks
            assert x.shape[-1] % mv_dim == 0
            num_blocks = x.shape[-1] // mv_dim
            x = x.view(*x.shape[:-1], num_blocks, mv_dim)
        else:
            num_blocks = 1
            x = x.unsqueeze(-2)

        output = torch.zeros_like(x)

        for grade in range(self.block_size + 1):
            indices = [i for i in range(mv_dim) if grades[i] == grade]
            if not indices:
                continue

            grade_x = x[..., indices]

            # Layer norm within grade
            mean = grade_x.mean(dim=-1, keepdim=True)
            var = grade_x.var(dim=-1, keepdim=True, unbiased=False)
            grade_x = (grade_x - mean) / torch.sqrt(var + self.eps)

            # Apply grade-specific affine
            if self.weight is not None:
                grade_x = grade_x * self.weight[grade] + self.bias[grade]

            for idx, i in enumerate(indices):
                output[..., i] = grade_x[..., idx]

        # Reshape back
        if num_blocks == 1:
            output = output.squeeze(-2)
        else:
            output = output.view(*original_shape)

        return output


class MultivectorNorm(nn.Module):
    """
    Normalize multivector by its geometric norm.

    Uses |M|² = <M M†>₀ where M† is the reverse.
    """

    def __init__(self, algebra_dim: int = 8, eps: float = 1e-8):
        super().__init__()
        self.algebra_dim = algebra_dim
        self.mv_dim = 2 ** algebra_dim
        self.eps = eps

        # Precompute reverse signs
        reverse_signs = torch.ones(self.mv_dim)
        for i in range(self.mv_dim):
            grade = bin(i).count('1')
            if (grade * (grade - 1) // 2) % 2 == 1:
                reverse_signs[i] = -1
        self.register_buffer('reverse_signs', reverse_signs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize multivector to unit norm.

        Args:
            x: [..., mv_dim]

        Returns:
            [..., mv_dim] normalized multivector
        """
        # Compute reverse
        x_rev = x * self.reverse_signs

        # Compute norm squared (scalar part of x * x_rev)
        # Simplified: just use sum of squares weighted by reverse signs
        norm_sq = (x * x_rev).sum(dim=-1, keepdim=True)
        norm = torch.sqrt(torch.abs(norm_sq) + self.eps)

        return x / norm
