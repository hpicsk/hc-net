"""
Clifford Algebra implementation with precomputed Cayley tables.

This module implements Cl(d,0) - Clifford algebra with Euclidean signature.
Each basis element e_i satisfies e_i^2 = +1.

Key insight: Basis elements are indexed by subsets of {0,1,...,d-1}.
- Index 0 = scalar (empty set)
- Index 1 = e_0, Index 2 = e_1, Index 4 = e_2, etc. (single elements)
- Index 3 = e_0 e_1, Index 5 = e_0 e_2, etc. (pairs = bivectors)
"""

import torch
import numpy as np
from functools import lru_cache
from typing import Tuple, Optional


class CliffordAlgebra:
    """
    Clifford Algebra Cl(d, 0) with precomputed multiplication structure.

    Attributes:
        d: Number of basis vectors (generators)
        dim: Total algebra dimension = 2^d
        cayley_table: [dim, dim, dim] sparse sign tensor for geometric product
        grades: [dim] grade of each basis element
        grade_masks: Dict[int, Tensor] boolean mask for each grade
    """

    def __init__(self, d: int, device: str = 'cuda'):
        """
        Initialize Clifford algebra Cl(d, 0).

        Args:
            d: Number of generators (basis vectors)
            device: Device for tensors ('cuda' or 'cpu')
        """
        self.d = d
        self.dim = 2 ** d
        self.device = device

        # Build Cayley table and grade structure
        self._cayley_indices, self._cayley_signs = self._build_cayley_table()
        self.grades = self._compute_grades()
        self.grade_masks = self._build_grade_masks()

        # Move to device
        self._cayley_indices = self._cayley_indices.to(device)
        self._cayley_signs = self._cayley_signs.to(device)
        self.grades = self.grades.to(device)
        for k in self.grade_masks:
            self.grade_masks[k] = self.grade_masks[k].to(device)

    def _build_cayley_table(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build the Cayley table for geometric product.

        For basis elements e_I and e_J (indexed by subsets I, J):
        e_I * e_J = sign(I, J) * e_{I Δ J}

        where I Δ J is symmetric difference and sign comes from
        counting transpositions to order the product.

        Returns:
            indices: [dim, dim] result index for each product
            signs: [dim, dim] sign (+1 or -1) for each product
        """
        dim = self.dim
        d = self.d

        indices = torch.zeros(dim, dim, dtype=torch.long)
        signs = torch.zeros(dim, dim, dtype=torch.float32)

        for i in range(dim):
            for j in range(dim):
                # i and j are bit representations of subsets
                result_idx, sign = self._multiply_basis(i, j)
                indices[i, j] = result_idx
                signs[i, j] = sign

        return indices, signs

    def _multiply_basis(self, i: int, j: int) -> Tuple[int, int]:
        """
        Multiply two basis elements represented as bit patterns.

        e_I * e_J where I, J are the sets of indices in i, j.

        Args:
            i: Bit pattern for first basis element
            j: Bit pattern for second basis element

        Returns:
            (result_index, sign): Index and sign of the product
        """
        # Result basis element is symmetric difference
        result = i ^ j

        # Count sign: number of transpositions needed
        # For each bit in j, count how many bits in i are to its right
        sign = 1

        # Extract indices from i and j
        i_indices = self._bits_to_indices(i)
        j_indices = self._bits_to_indices(j)

        # Count inversions: for each index in j, how many larger indices in i?
        inversions = 0
        for jk in j_indices:
            for ik in i_indices:
                if ik > jk:
                    inversions += 1

        # Also count e_k * e_k = 1 contributions (Euclidean signature)
        # Intersection elements each contribute e_k^2 = 1
        # No sign change for Cl(d,0)

        # Additionally, we need to account for moving j through i
        # Each swap of adjacent elements changes sign
        # Total swaps = sum over j_k of |{i_m : i_m > j_k}|

        if inversions % 2 == 1:
            sign = -1

        return result, sign

    def _bits_to_indices(self, n: int) -> list:
        """Convert bit pattern to list of set indices."""
        indices = []
        pos = 0
        while n:
            if n & 1:
                indices.append(pos)
            n >>= 1
            pos += 1
        return indices

    def _compute_grades(self) -> torch.Tensor:
        """Compute grade (number of basis vectors) for each element."""
        grades = torch.zeros(self.dim, dtype=torch.long)
        for i in range(self.dim):
            grades[i] = bin(i).count('1')
        return grades

    def _build_grade_masks(self) -> dict:
        """Build boolean masks for each grade."""
        masks = {}
        for grade in range(self.d + 1):
            mask = self.grades == grade
            masks[grade] = mask
        return masks

    def geometric_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute geometric product of multivectors x and y.

        Args:
            x: [..., dim] first multivector
            y: [..., dim] second multivector

        Returns:
            [..., dim] geometric product x * y
        """
        # Use advanced indexing for batched computation
        # x[..., i] * y[..., j] contributes to result[..., indices[i,j]] with sign signs[i,j]

        # Expand for broadcasting: x[..., i, 1] * y[..., 1, j]
        x_exp = x.unsqueeze(-1)  # [..., dim, 1]
        y_exp = y.unsqueeze(-2)  # [..., 1, dim]

        # Compute all products
        products = x_exp * y_exp  # [..., dim, dim]

        # Apply signs
        products = products * self._cayley_signs  # [..., dim, dim]

        # Scatter-add to result indices
        batch_shape = x.shape[:-1]
        result = torch.zeros(*batch_shape, self.dim, device=x.device, dtype=x.dtype)

        # Flatten batch dimensions for scatter
        flat_products = products.reshape(-1, self.dim, self.dim)
        flat_result = result.reshape(-1, self.dim)

        for i in range(self.dim):
            for j in range(self.dim):
                k = self._cayley_indices[i, j].item()
                flat_result[:, k] += flat_products[:, i, j]

        return flat_result.reshape(*batch_shape, self.dim)

    def geometric_product_optimized(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Optimized geometric product using einsum with dense Cayley tensor.

        For small d (d <= 8), we can afford to store the full [dim, dim, dim] tensor.
        """
        target_device = x.device
        if not hasattr(self, '_cayley_dense') or self._cayley_dense.device != target_device:
            # Build dense Cayley tensor (or rebuild if device changed)
            self._cayley_dense = torch.zeros(
                self.dim, self.dim, self.dim,
                device=target_device, dtype=torch.float32
            )
            indices_cpu = self._cayley_indices.cpu()
            signs_cpu = self._cayley_signs.cpu()
            for i in range(self.dim):
                for j in range(self.dim):
                    k = indices_cpu[i, j].item()
                    self._cayley_dense[i, j, k] = signs_cpu[i, j]

        # Batched geometric product via einsum
        return torch.einsum('...i,...j,ijk->...k', x, y, self._cayley_dense)

    def grade_projection(self, x: torch.Tensor, grade: int) -> torch.Tensor:
        """
        Project multivector onto a specific grade.

        Args:
            x: [..., dim] multivector
            grade: Grade to project onto (0=scalar, 1=vector, 2=bivector, ...)

        Returns:
            [..., dim] projected multivector (other grades zeroed)
        """
        mask = self.grade_masks.get(grade)
        if mask is None:
            return torch.zeros_like(x)
        return x * mask.to(device=x.device, dtype=x.dtype)

    def scalar_part(self, x: torch.Tensor) -> torch.Tensor:
        """Extract scalar (grade-0) component."""
        return x[..., 0:1]

    def vector_part(self, x: torch.Tensor) -> torch.Tensor:
        """Extract vector (grade-1) components."""
        return self.grade_projection(x, 1)

    def bivector_part(self, x: torch.Tensor) -> torch.Tensor:
        """Extract bivector (grade-2) components."""
        return self.grade_projection(x, 2)

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the norm of a multivector.

        Uses |x|^2 = <x * x_rev>_0 (scalar part of x times its reverse)
        """
        x_rev = self.reverse(x)
        product = self.geometric_product_optimized(x, x_rev)
        return torch.sqrt(torch.abs(product[..., 0:1]) + 1e-8)

    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the reverse (reversion) of a multivector.

        Reverse of e_{i1}...e_{ik} = e_{ik}...e_{i1} = (-1)^{k(k-1)/2} e_{i1}...e_{ik}
        """
        signs = torch.ones(self.dim, device=x.device, dtype=x.dtype)
        grades_local = self.grades.to(x.device)
        for i in range(self.dim):
            grade = grades_local[i].item()
            if (grade * (grade - 1) // 2) % 2 == 1:
                signs[i] = -1
        return x * signs

    def embed_vector(self, v: torch.Tensor) -> torch.Tensor:
        """
        Embed a d-dimensional vector into the algebra.

        Args:
            v: [..., d] vector

        Returns:
            [..., dim] multivector with vector components set
        """
        batch_shape = v.shape[:-1]
        result = torch.zeros(*batch_shape, self.dim, device=v.device, dtype=v.dtype)

        # Vector components are at indices 2^0, 2^1, ..., 2^(d-1)
        for i in range(self.d):
            result[..., 2**i] = v[..., i]

        return result

    def extract_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the vector part as a d-dimensional vector.

        Args:
            x: [..., dim] multivector

        Returns:
            [..., d] vector components
        """
        batch_shape = x.shape[:-1]
        result = torch.zeros(*batch_shape, self.d, device=x.device, dtype=x.dtype)

        for i in range(self.d):
            result[..., i] = x[..., 2**i]

        return result


# Cached algebra instances
@lru_cache(maxsize=8)
def get_algebra(d: int, device: str = 'cuda') -> CliffordAlgebra:
    """Get cached CliffordAlgebra instance."""
    return CliffordAlgebra(d, device)
