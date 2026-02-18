"""
Optimized Clifford algebra operations for neural networks.

These functions provide GPU-accelerated batched operations
using the precomputed Cayley tables from CliffordAlgebra.
"""

import torch
import torch.nn.functional as F
from typing import Optional
from .clifford import CliffordAlgebra, get_algebra


def geometric_product(
    x: torch.Tensor,
    y: torch.Tensor,
    algebra: CliffordAlgebra
) -> torch.Tensor:
    """
    Batched geometric product using optimized einsum.

    Args:
        x: [..., dim] first multivector
        y: [..., dim] second multivector
        algebra: CliffordAlgebra instance

    Returns:
        [..., dim] geometric product
    """
    return algebra.geometric_product_optimized(x, y)


def grade_projection(
    x: torch.Tensor,
    grade: int,
    algebra: CliffordAlgebra
) -> torch.Tensor:
    """
    Project multivector onto specific grade.

    Args:
        x: [..., dim] multivector
        grade: Target grade (0, 1, 2, ...)
        algebra: CliffordAlgebra instance

    Returns:
        [..., dim] projected multivector
    """
    return algebra.grade_projection(x, grade)


def outer_product(
    x: torch.Tensor,
    y: torch.Tensor,
    algebra: CliffordAlgebra
) -> torch.Tensor:
    """
    Outer (wedge) product of multivectors.

    x ∧ y extracts the antisymmetric part of the geometric product.
    For vectors: x ∧ y = (xy - yx) / 2

    Args:
        x: [..., dim] first multivector
        y: [..., dim] second multivector
        algebra: CliffordAlgebra instance

    Returns:
        [..., dim] outer product
    """
    xy = geometric_product(x, y, algebra)
    yx = geometric_product(y, x, algebra)

    # For pure grade-k and grade-l inputs, wedge product is grade-(k+l) part
    # General formula: project onto higher grades
    result = torch.zeros_like(xy)

    # Sum contributions from grades higher than either input's max grade
    for grade in range(algebra.d + 1):
        grade_part = algebra.grade_projection(xy, grade)
        # The outer product is (xy + (-1)^(grade_x * grade_y) yx) / 2
        # For simplicity, use antisymmetric part
        result += algebra.grade_projection((xy - yx) / 2, grade)

    return result


def inner_product(
    x: torch.Tensor,
    y: torch.Tensor,
    algebra: CliffordAlgebra
) -> torch.Tensor:
    """
    Inner (dot) product of multivectors.

    For vectors: x · y = (xy + yx) / 2 = scalar

    Args:
        x: [..., dim] first multivector
        y: [..., dim] second multivector
        algebra: CliffordAlgebra instance

    Returns:
        [..., dim] inner product
    """
    xy = geometric_product(x, y, algebra)
    yx = geometric_product(y, x, algebra)

    # Inner product extracts lower grades
    return (xy + yx) / 2


def sandwich_product(
    x: torch.Tensor,
    rotor: torch.Tensor,
    algebra: CliffordAlgebra
) -> torch.Tensor:
    """
    Sandwich product: R x R†

    This performs a rotation/reflection of x by rotor R.

    Args:
        x: [..., dim] multivector to transform
        rotor: [..., dim] rotor (unit multivector)
        algebra: CliffordAlgebra instance

    Returns:
        [..., dim] transformed multivector
    """
    rotor_rev = algebra.reverse(rotor)
    return geometric_product(
        geometric_product(rotor, x, algebra),
        rotor_rev,
        algebra
    )


def multivector_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    algebra: CliffordAlgebra
) -> torch.Tensor:
    """
    Linear transformation in multivector space.

    Computes: sum_j W_ij * x_j (geometric product)

    Args:
        x: [B, in_features, dim] input multivectors
        weight: [out_features, in_features, dim] weight multivectors
        bias: [out_features, dim] optional bias multivector
        algebra: CliffordAlgebra instance

    Returns:
        [B, out_features, dim] output multivectors
    """
    B, in_f, dim = x.shape
    out_f = weight.shape[0]

    # Expand for broadcasting
    # x: [B, 1, in_f, dim]
    # weight: [1, out_f, in_f, dim]
    x_exp = x.unsqueeze(1)
    w_exp = weight.unsqueeze(0)

    # Compute pairwise geometric products: [B, out_f, in_f, dim]
    products = torch.zeros(B, out_f, in_f, dim, device=x.device, dtype=x.dtype)

    for b in range(B):
        for o in range(out_f):
            for i in range(in_f):
                products[b, o, i] = algebra.geometric_product_optimized(
                    x[b, i].unsqueeze(0),
                    weight[o, i].unsqueeze(0)
                ).squeeze(0)

    # Sum over input features
    result = products.sum(dim=2)  # [B, out_f, dim]

    if bias is not None:
        result = result + bias.unsqueeze(0)

    return result


def multivector_linear_fast(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    cayley: torch.Tensor
) -> torch.Tensor:
    """
    Fast linear transformation using batched einsum.

    Args:
        x: [B, in_features, dim] input multivectors
        weight: [out_features, in_features, dim] weight multivectors
        bias: [out_features, dim] optional bias
        cayley: [dim, dim, dim] dense Cayley table

    Returns:
        [B, out_features, dim] output multivectors
    """
    # Geometric product via einsum: x[b,i,a] * w[o,i,b] * cayley[a,b,c] -> [b,o,i,c]
    products = torch.einsum('bia,oib,abc->boic', x, weight, cayley)

    # Sum over input features
    result = products.sum(dim=2)  # [B, out_f, dim]

    if bias is not None:
        result = result + bias.unsqueeze(0)

    return result


def block_geometric_product(
    x: torch.Tensor,
    y: torch.Tensor,
    block_size: int,
    algebra: CliffordAlgebra
) -> torch.Tensor:
    """
    Block-wise geometric product for high-dimensional feature maps.

    Splits channels into blocks and computes geometric product within each block.

    Args:
        x: [B, C, H, W] feature map
        y: [B, C, H, W] feature map (or filter)
        block_size: Number of channels per block (d in Cl(d,0))
        algebra: CliffordAlgebra instance for the block

    Returns:
        [B, C', H, W] where C' may differ due to grade expansion
    """
    B, C, H, W = x.shape
    assert C % block_size == 0, f"Channels {C} must be divisible by block_size {block_size}"

    num_blocks = C // block_size

    # Reshape to blocks: [B, num_blocks, block_size, H, W]
    x_blocks = x.view(B, num_blocks, block_size, H, W)
    y_blocks = y.view(B, num_blocks, block_size, H, W)

    # Embed each block's channels as vector in Clifford algebra
    # block_size channels -> vector in Cl(block_size, 0)
    # Then compute geometric product

    # For each spatial location, treat the block_size channels as a vector
    # Embed into full multivector space
    x_mv = embed_channels_to_multivector(x_blocks, algebra)  # [B, num_blocks, 2^d, H, W]
    y_mv = embed_channels_to_multivector(y_blocks, algebra)

    # Geometric product per block per spatial location
    # Reshape for batch processing: [B * num_blocks * H * W, 2^d]
    x_flat = x_mv.permute(0, 1, 3, 4, 2).reshape(-1, algebra.dim)
    y_flat = y_mv.permute(0, 1, 3, 4, 2).reshape(-1, algebra.dim)

    result_flat = algebra.geometric_product_optimized(x_flat, y_flat)

    # Reshape back
    result = result_flat.reshape(B, num_blocks, H, W, algebra.dim)
    result = result.permute(0, 1, 4, 2, 3)  # [B, num_blocks, 2^d, H, W]

    return result


def embed_channels_to_multivector(
    x: torch.Tensor,
    algebra: CliffordAlgebra
) -> torch.Tensor:
    """
    Embed block channels as vectors in Clifford algebra.

    Args:
        x: [B, num_blocks, d, H, W] where d = algebra.d

    Returns:
        [B, num_blocks, 2^d, H, W] multivector representation
    """
    B, num_blocks, d, H, W = x.shape
    assert d == algebra.d, f"Channel dimension {d} must match algebra dimension {algebra.d}"

    dim = algebra.dim
    result = torch.zeros(B, num_blocks, dim, H, W, device=x.device, dtype=x.dtype)

    # Set vector components: channel i -> basis element 2^i
    for i in range(d):
        result[:, :, 2**i, :, :] = x[:, :, i, :, :]

    return result


def extract_multivector_to_channels(
    x: torch.Tensor,
    algebra: CliffordAlgebra,
    mode: str = 'vector'
) -> torch.Tensor:
    """
    Extract channels from multivector representation.

    Args:
        x: [B, num_blocks, 2^d, H, W] multivector representation
        algebra: CliffordAlgebra instance
        mode: 'vector' (grade-1 only), 'all' (all components),
              'scalar_vector' (grades 0 and 1)

    Returns:
        [B, C, H, W] where C depends on mode
    """
    B, num_blocks, dim, H, W = x.shape
    d = algebra.d

    if mode == 'vector':
        # Extract only vector components
        result = torch.zeros(B, num_blocks, d, H, W, device=x.device, dtype=x.dtype)
        for i in range(d):
            result[:, :, i, :, :] = x[:, :, 2**i, :, :]
        # Reshape: [B, num_blocks * d, H, W]
        return result.view(B, num_blocks * d, H, W)

    elif mode == 'all':
        # Return all multivector components
        return x.view(B, num_blocks * dim, H, W)

    elif mode == 'scalar_vector':
        # Return scalar + vector components
        num_components = 1 + d
        result = torch.zeros(B, num_blocks, num_components, H, W,
                           device=x.device, dtype=x.dtype)
        result[:, :, 0, :, :] = x[:, :, 0, :, :]  # scalar
        for i in range(d):
            result[:, :, 1+i, :, :] = x[:, :, 2**i, :, :]
        return result.view(B, num_blocks * num_components, H, W)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_bivector_magnitude(
    x: torch.Tensor,
    algebra: CliffordAlgebra
) -> torch.Tensor:
    """
    Compute magnitude of bivector (grade-2) components.

    This measures the "interaction strength" between features.

    Args:
        x: [..., dim] multivector

    Returns:
        [..., 1] bivector magnitude
    """
    bivector = algebra.grade_projection(x, 2)
    return torch.sqrt((bivector ** 2).sum(dim=-1, keepdim=True) + 1e-8)
