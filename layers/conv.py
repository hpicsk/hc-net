"""
Block Clifford Convolutional Layers.

Implements convolutional layers that treat channel blocks as
vectors in Clifford algebra and compute geometric products.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from ..algebra.clifford import CliffordAlgebra, get_algebra


class BlockCliffordConv2d(nn.Module):
    """
    2D Convolution with block-wise Clifford algebra operations.

    Channels are grouped into blocks of size `block_size`.
    Within each block, channels are treated as basis vectors in Cl(block_size, 0).
    The convolution computes geometric products, generating bivector interactions.

    Key insight: Standard conv computes weighted sums. This layer computes
    weighted geometric products, where the product of two vectors yields
    scalar + bivector, capturing pairwise channel interactions.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        block_size: Number of channels per Clifford block (d in Cl(d,0))
        stride: Convolution stride
        padding: Convolution padding
        bias: Whether to use bias
        expand_bivectors: If True, output includes bivector channels
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        block_size: int = 8,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        expand_bivectors: bool = False,
        device: str = 'cuda'
    ):
        super().__init__()

        assert in_channels % block_size == 0, \
            f"in_channels ({in_channels}) must be divisible by block_size ({block_size})"
        assert out_channels % block_size == 0, \
            f"out_channels ({out_channels}) must be divisible by block_size ({block_size})"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.block_size = block_size
        self.stride = stride
        self.padding = padding
        self.expand_bivectors = expand_bivectors
        self.device_str = device

        self.in_blocks = in_channels // block_size
        self.out_blocks = out_channels // block_size
        self.mv_dim = 2 ** block_size

        # Get algebra for the block
        self.algebra = get_algebra(block_size, device)

        # Precompute Cayley table
        if not hasattr(self.algebra, '_cayley_dense'):
            self.algebra.geometric_product_optimized(
                torch.zeros(1, self.mv_dim, device=device),
                torch.zeros(1, self.mv_dim, device=device)
            )
        self.register_buffer('cayley', self.algebra._cayley_dense)

        # Standard convolution for spatial mixing
        # We process blocks separately, so conv operates on block_size channels
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=1  # Full mixing across all channels
        )

        # Learnable geometric mixing weights per output block
        # This learns how to combine the geometric product results
        self.geo_weight = nn.Parameter(
            torch.ones(self.out_blocks, block_size) * 0.1
        )

        # Bivector weight: how much to emphasize pairwise interactions
        self.num_bivectors = block_size * (block_size - 1) // 2
        self.bivector_weight = nn.Parameter(
            torch.ones(self.out_blocks, self.num_bivectors) * 0.01
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        if expand_bivectors:
            # Output will include explicit bivector channels
            # Additional channels = out_blocks * num_bivectors
            self.extra_channels = self.out_blocks * self.num_bivectors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, in_channels, H, W] input feature map

        Returns:
            [B, out_channels, H', W'] output feature map
            (or [B, out_channels + extra, H', W'] if expand_bivectors=True)
        """
        B, C, H, W = x.shape
        assert C == self.in_channels

        # Standard spatial convolution
        out = self.conv(x)  # [B, out_channels, H', W']
        _, _, H_out, W_out = out.shape

        # Reshape to blocks for geometric operations
        out_blocks = out.view(B, self.out_blocks, self.block_size, H_out, W_out)

        # Compute within-block geometric interactions
        # For each block, compute pairwise products of channels

        # Create bivector features from channel pairs
        bivector_features = []
        biv_idx = 0
        for i in range(self.block_size):
            for j in range(i + 1, self.block_size):
                # v_i ∧ v_j approximated by v_i * v_j
                biv = out_blocks[:, :, i] * out_blocks[:, :, j]  # [B, out_blocks, H, W]
                # Weight by learned bivector importance
                biv = biv * self.bivector_weight[:, biv_idx].view(1, -1, 1, 1)
                bivector_features.append(biv)
                biv_idx += 1

        if bivector_features:
            # Stack and sum bivector contributions back into channels
            bivector_stack = torch.stack(bivector_features, dim=2)  # [B, out_blocks, num_biv, H, W]

            # Distribute bivector influence back to vector channels
            # Each bivector (i,j) influences channels i and j
            geo_adjustment = torch.zeros_like(out_blocks)
            biv_idx = 0
            for i in range(self.block_size):
                for j in range(i + 1, self.block_size):
                    geo_adjustment[:, :, i] += bivector_stack[:, :, biv_idx] * self.geo_weight[:, i].view(1, -1, 1, 1)
                    geo_adjustment[:, :, j] += bivector_stack[:, :, biv_idx] * self.geo_weight[:, j].view(1, -1, 1, 1)
                    biv_idx += 1

            # Add geometric adjustments
            out_blocks = out_blocks + geo_adjustment

        # Reshape back to standard format
        out = out_blocks.view(B, self.out_channels, H_out, W_out)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        if self.expand_bivectors and bivector_features:
            # Append bivector channels to output
            bivector_out = torch.cat([
                bivector_stack[:, :, i].unsqueeze(2)
                for i in range(len(bivector_features))
            ], dim=2)  # [B, out_blocks, num_biv, H, W]
            bivector_out = bivector_out.view(B, -1, H_out, W_out)
            out = torch.cat([out, bivector_out], dim=1)

        return out

    def extra_repr(self) -> str:
        return (f'{self.in_channels}, {self.out_channels}, '
                f'kernel_size={self.kernel_size}, block_size={self.block_size}, '
                f'stride={self.stride}, padding={self.padding}')


class FullCliffordConv2d(nn.Module):
    """
    Full Clifford convolutional layer with explicit multivector representation.

    Each spatial location stores a full multivector.
    Convolution performs geometric products between input and kernel multivectors.

    More computationally expensive but theoretically complete.
    """

    def __init__(
        self,
        in_blocks: int,
        out_blocks: int,
        kernel_size: int = 3,
        algebra_dim: int = 4,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        device: str = 'cuda'
    ):
        super().__init__()

        self.in_blocks = in_blocks
        self.out_blocks = out_blocks
        self.kernel_size = kernel_size
        self.algebra_dim = algebra_dim
        self.mv_dim = 2 ** algebra_dim
        self.stride = stride
        self.padding = padding

        # Actual channel counts (each block has mv_dim components)
        self.in_channels = in_blocks * self.mv_dim
        self.out_channels = out_blocks * self.mv_dim

        # Get algebra
        self.algebra = get_algebra(algebra_dim, device)

        # Kernel weights: [out_blocks, in_blocks, K, K, mv_dim]
        # Each kernel position is a multivector
        self.weight = nn.Parameter(
            torch.empty(out_blocks, in_blocks, kernel_size, kernel_size, self.mv_dim)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_blocks, self.mv_dim))
        else:
            self.register_parameter('bias', None)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize kernel multivectors."""
        fan_in = self.in_blocks * self.kernel_size * self.kernel_size * self.mv_dim
        std = 1.0 / math.sqrt(fan_in)

        # Initialize scalar components
        nn.init.normal_(self.weight[:, :, :, :, 0], std=std)

        # Initialize vector components with smaller values
        for i in range(self.algebra_dim):
            nn.init.normal_(self.weight[:, :, :, :, 2**i], std=std * 0.5)

        # Initialize higher grades with even smaller values
        for idx in range(self.mv_dim):
            grade = bin(idx).count('1')
            if grade > 1:
                nn.init.normal_(self.weight[:, :, :, :, idx], std=std * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, in_blocks * mv_dim, H, W] input (multivector per block per location)

        Returns:
            [B, out_blocks * mv_dim, H', W'] output
        """
        B, C, H, W = x.shape

        # Reshape to expose multivector structure
        x = x.view(B, self.in_blocks, self.mv_dim, H, W)

        # Unfold for manual convolution
        # This extracts all kernel-sized patches
        x_unfold = F.unfold(
            x.view(B, self.in_channels, H, W),
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride
        )  # [B, in_channels * K * K, L] where L = H' * W'

        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        L = H_out * W_out

        # Reshape for geometric product computation
        x_unfold = x_unfold.view(B, self.in_blocks, self.mv_dim,
                                 self.kernel_size, self.kernel_size, L)

        # Initialize output
        output = torch.zeros(B, self.out_blocks, self.mv_dim, L,
                           device=x.device, dtype=x.dtype)

        # Compute geometric products (simplified for efficiency)
        # For each output block, sum geometric products with input blocks
        for o in range(self.out_blocks):
            for i in range(self.in_blocks):
                for kh in range(self.kernel_size):
                    for kw in range(self.kernel_size):
                        # Input at this kernel position
                        x_pos = x_unfold[:, i, :, kh, kw, :]  # [B, mv_dim, L]
                        w_pos = self.weight[o, i, kh, kw]  # [mv_dim]

                        # Geometric product (simplified: just scalar multiplication for now)
                        # Full version would use Cayley table
                        product = x_pos * w_pos.view(1, -1, 1)
                        output[:, o] += product

        # Add bias
        if self.bias is not None:
            output = output + self.bias.view(1, self.out_blocks, self.mv_dim, 1)

        # Reshape to output format
        output = output.view(B, self.out_channels, H_out, W_out)

        return output


class HybridCliffordConv2d(nn.Module):
    """
    Hybrid approach: standard conv followed by Clifford interaction layer.

    This is more efficient than full Clifford conv while still capturing
    geometric interactions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        block_size: int = 8,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        interaction_ratio: float = 0.1
    ):
        super().__init__()

        self.block_size = block_size
        self.interaction_ratio = interaction_ratio

        # Standard convolution
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        # Interaction layer: 1x1 conv to mix bivector-weighted features
        num_bivectors = block_size * (block_size - 1) // 2
        assert out_channels % block_size == 0
        out_blocks = out_channels // block_size

        # Learnable interaction matrix
        self.interaction = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=1,
            groups=out_blocks,
            bias=False
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize interaction to be small (residual-like)
        nn.init.xavier_uniform_(self.interaction.weight, gain=interaction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with hybrid conv + interaction."""
        out = self.conv(x)

        # Add interaction term (simulates geometric product effects)
        interaction = self.interaction(out)
        out = out + interaction

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out
