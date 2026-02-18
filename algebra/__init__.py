"""
Clifford Algebra module for HC-Net.

Provides the core algebraic structures:
- CliffordAlgebra: Precomputed Cayley tables and grade structures
- Batched geometric product operations optimized for GPU
"""

from .clifford import CliffordAlgebra, get_algebra
from .operations import geometric_product, grade_projection, outer_product, inner_product

__all__ = [
    'CliffordAlgebra',
    'get_algebra',
    'geometric_product',
    'grade_projection',
    'outer_product',
    'inner_product',
]
