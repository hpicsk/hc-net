"""
Activation functions for Clifford algebra neural networks.

Provides grade-aware nonlinearities that respect the
geometric structure of multivectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class CliffordReLU(nn.Module):
    """
    Grade-aware ReLU activation.

    Applies ReLU to each grade separately, optionally with
    different behaviors for different grades.

    Args:
        algebra_dim: Dimension of Clifford algebra generators
        mode: 'all' (ReLU on all), 'scalar' (ReLU on scalar only),
              'magnitude' (ReLU on magnitude, preserve direction)
    """

    def __init__(self, algebra_dim: int = 8, mode: str = 'all'):
        super().__init__()
        self.algebra_dim = algebra_dim
        self.mv_dim = 2 ** algebra_dim
        self.mode = mode

        # Precompute grade information
        self.grades = [bin(i).count('1') for i in range(self.mv_dim)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [..., mv_dim] multivector

        Returns:
            [..., mv_dim] activated multivector
        """
        if self.mode == 'all':
            return F.relu(x)

        elif self.mode == 'scalar':
            # Only apply ReLU to scalar component
            out = x.clone()
            out[..., 0] = F.relu(x[..., 0])
            return out

        elif self.mode == 'magnitude':
            # Preserve direction, apply ReLU to magnitude
            norm = torch.sqrt((x ** 2).sum(dim=-1, keepdim=True) + 1e-8)
            direction = x / norm
            activated_norm = F.relu(norm)
            return direction * activated_norm

        elif self.mode == 'grade_wise':
            # Apply ReLU to each grade's magnitude
            out = torch.zeros_like(x)

            for grade in range(self.algebra_dim + 1):
                indices = [i for i, g in enumerate(self.grades) if g == grade]
                if not indices:
                    continue

                grade_x = x[..., indices]
                grade_norm = torch.sqrt((grade_x ** 2).sum(dim=-1, keepdim=True) + 1e-8)
                grade_dir = grade_x / grade_norm
                activated = grade_dir * F.relu(grade_norm)

                for idx, i in enumerate(indices):
                    out[..., i] = activated[..., idx]

            return out

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class CliffordGELU(nn.Module):
    """
    GELU activation for multivectors.

    Smoother alternative to ReLU.
    """

    def __init__(self, algebra_dim: int = 8, mode: str = 'all'):
        super().__init__()
        self.algebra_dim = algebra_dim
        self.mv_dim = 2 ** algebra_dim
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GELU activation."""
        if self.mode == 'all':
            return F.gelu(x)

        elif self.mode == 'magnitude':
            norm = torch.sqrt((x ** 2).sum(dim=-1, keepdim=True) + 1e-8)
            direction = x / norm
            activated_norm = F.gelu(norm)
            return direction * activated_norm

        else:
            return F.gelu(x)


class MVSiLU(nn.Module):
    """
    Multivector SiLU (Swish) activation.

    SiLU(x) = x * sigmoid(x)

    For multivectors, we can apply this component-wise or
    using the scalar to gate all components.
    """

    def __init__(self, algebra_dim: int = 8, mode: str = 'scalar_gate'):
        super().__init__()
        self.algebra_dim = algebra_dim
        self.mv_dim = 2 ** algebra_dim
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SiLU activation."""
        if self.mode == 'all':
            return F.silu(x)

        elif self.mode == 'scalar_gate':
            # Use scalar component to gate all components
            scalar = x[..., 0:1]
            gate = torch.sigmoid(scalar)
            return x * gate

        elif self.mode == 'magnitude_gate':
            # Use magnitude to gate
            norm = torch.sqrt((x ** 2).sum(dim=-1, keepdim=True) + 1e-8)
            gate = torch.sigmoid(norm)
            return x * gate

        else:
            return F.silu(x)


class GradeDropout(nn.Module):
    """
    Dropout that operates on entire grades.

    During training, randomly zeros out entire grades
    to encourage the network to use multiple geometric components.
    """

    def __init__(
        self,
        algebra_dim: int = 8,
        p: float = 0.1,
        grade_probs: Optional[List[float]] = None
    ):
        super().__init__()
        self.algebra_dim = algebra_dim
        self.mv_dim = 2 ** algebra_dim
        self.p = p

        # Different dropout probabilities per grade
        if grade_probs is None:
            self.grade_probs = [p] * (algebra_dim + 1)
        else:
            self.grade_probs = grade_probs

        # Precompute grade masks
        grades = [bin(i).count('1') for i in range(self.mv_dim)]
        self.grade_masks = {}
        for g in range(algebra_dim + 1):
            mask = torch.tensor([1.0 if grades[i] == g else 0.0 for i in range(self.mv_dim)])
            self.register_buffer(f'grade_mask_{g}', mask)
            self.grade_masks[g] = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply grade dropout."""
        if not self.training:
            return x

        # Generate dropout mask per grade
        device = x.device
        dropout_mask = torch.ones(self.mv_dim, device=device)

        for grade, prob in enumerate(self.grade_probs):
            if torch.rand(1).item() < prob:
                # Zero out this grade
                grade_mask = getattr(self, f'grade_mask_{grade}').to(device)
                dropout_mask = dropout_mask * (1 - grade_mask)

        # Apply mask
        return x * dropout_mask


class SpinorActivation(nn.Module):
    """
    Activation that preserves spinor structure.

    Spinors are elements of the even subalgebra (grades 0, 2, 4, ...).
    This activation preserves the spinor property.
    """

    def __init__(self, algebra_dim: int = 8):
        super().__init__()
        self.algebra_dim = algebra_dim
        self.mv_dim = 2 ** algebra_dim

        # Compute even grade mask
        even_mask = torch.zeros(self.mv_dim)
        for i in range(self.mv_dim):
            if bin(i).count('1') % 2 == 0:
                even_mask[i] = 1.0
        self.register_buffer('even_mask', even_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spinor-preserving activation.

        Projects to even subalgebra and applies nonlinearity.
        """
        # Project to even grades
        x_even = x * self.even_mask

        # Apply activation to even components
        return F.gelu(x_even)


class RotorNormalization(nn.Module):
    """
    Normalize to unit rotor (for rotation representations).

    Rotors are elements R such that R R† = 1.
    This normalizes to satisfy this constraint.
    """

    def __init__(self, algebra_dim: int = 8, eps: float = 1e-8):
        super().__init__()
        self.algebra_dim = algebra_dim
        self.mv_dim = 2 ** algebra_dim
        self.eps = eps

        # Reverse signs
        reverse_signs = torch.ones(self.mv_dim)
        for i in range(self.mv_dim):
            grade = bin(i).count('1')
            if (grade * (grade - 1) // 2) % 2 == 1:
                reverse_signs[i] = -1
        self.register_buffer('reverse_signs', reverse_signs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to unit rotor."""
        # Compute x * x†
        x_rev = x * self.reverse_signs
        # For proper rotor norm, we'd compute full geometric product
        # Simplified: use L2 norm of even components
        even_mask = torch.zeros(self.mv_dim, device=x.device)
        for i in range(self.mv_dim):
            if bin(i).count('1') % 2 == 0:
                even_mask[i] = 1.0

        x_even = x * even_mask
        norm = torch.sqrt((x_even ** 2).sum(dim=-1, keepdim=True) + self.eps)
        return x / norm
