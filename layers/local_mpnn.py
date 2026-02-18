"""
Local Message Passing Neural Network (MPNN) module.

Provides distance-based message passing on a kNN graph with:
- Gaussian RBF distance encoding
- GPU-efficient kNN via torch.cdist + topk
- O(kN) = O(N) complexity since k is constant

Standalone PyTorch implementation with no external dependencies
beyond torch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class RadialBasisFunction(nn.Module):
    """
    Gaussian Radial Basis Function distance encoding.

    Encodes scalar distances into a vector of Gaussian basis function
    activations, providing a smooth, learnable distance representation.
    """

    def __init__(
        self,
        n_rbf: int = 20,
        cutoff: float = 5.0,
        trainable: bool = False,
    ):
        """
        Args:
            n_rbf: Number of RBF centers
            cutoff: Maximum distance (RBF centers span [0, cutoff])
            trainable: Whether RBF centers and widths are learnable
        """
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff

        # Centers evenly spaced in [0, cutoff]
        centers = torch.linspace(0, cutoff, n_rbf)
        # Width: inverse squared spacing
        widths = torch.full((n_rbf,), (cutoff / n_rbf) ** (-2))

        if trainable:
            self.centers = nn.Parameter(centers)
            self.widths = nn.Parameter(widths)
        else:
            self.register_buffer('centers', centers)
            self.register_buffer('widths', widths)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: [...] scalar distances

        Returns:
            [..., n_rbf] RBF activations
        """
        d = distances.unsqueeze(-1)  # [..., 1]
        return torch.exp(-self.widths * (d - self.centers) ** 2)


def compute_knn_edges(
    positions: torch.Tensor,
    k: int,
    cutoff: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute kNN edges from positions.

    Uses explicit pairwise distance computation (not torch.cdist) to
    support higher-order autograd (needed by energy-conserving models
    that compute forces via dE/dr).

    Args:
        positions: [B, N, 3] particle positions
        k: Number of nearest neighbors
        cutoff: Optional distance cutoff (edges beyond cutoff are masked)

    Returns:
        edge_idx: [B, N, k] indices of k nearest neighbors per node
        edge_dist: [B, N, k] distances to neighbors
        edge_mask: [B, N, k] boolean mask (True = valid edge)
    """
    B, N, _ = positions.shape
    k = min(k, N - 1)  # Can't have more neighbors than N-1

    # Pairwise distances via explicit ops (supports higher-order autograd,
    # unlike torch.cdist which lacks second-derivative support).
    diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [B, N, N, 3]
    dists = torch.sqrt((diff ** 2).sum(-1) + 1e-8)  # [B, N, N]

    # Set self-distance to inf to exclude self-loops
    eye = torch.eye(N, device=positions.device, dtype=torch.bool)
    dists_masked = dists.masked_fill(eye.unsqueeze(0), float('inf'))

    # Find k nearest neighbors (detach for index selection only)
    _, topk_idx = dists_masked.detach().topk(k, dim=-1, largest=False)

    # Gather distances using differentiable indexing
    topk_dists = torch.gather(dists_masked, dim=2, index=topk_idx)

    # Apply cutoff mask
    if cutoff is not None:
        edge_mask = topk_dists < cutoff
    else:
        edge_mask = torch.ones_like(topk_dists, dtype=torch.bool)

    return topk_idx, topk_dists, edge_mask


class LocalMPNNLayer(nn.Module):
    """
    Local message passing layer on kNN graph.

    For each node i:
    1. Find k nearest neighbors j
    2. Compute edge features: [h_i, h_j, RBF(d_ij)]
    3. Edge MLP: edge_features -> message
    4. Aggregate: sum messages per node
    5. Update: h_i' = LayerNorm(h_i + node_MLP(h_i || agg_msg))

    Complexity: O(kN) = O(N) since k is constant.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_rbf: int = 20,
        k_neighbors: int = 10,
        cutoff: float = 5.0,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: Hidden feature dimension
            n_rbf: Number of RBF centers for distance encoding
            k_neighbors: Number of nearest neighbors
            cutoff: Distance cutoff
            dropout: Dropout rate
        """
        super().__init__()

        self.k_neighbors = k_neighbors
        self.cutoff = cutoff

        # Distance encoding
        self.rbf = RadialBasisFunction(n_rbf=n_rbf, cutoff=cutoff)

        # Edge MLP: [h_i, h_j, rbf(d)] -> message
        edge_input_dim = hidden_dim * 2 + n_rbf
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Node update MLP: [h_i, agg_msg] -> update
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: [B, N, hidden_dim] node features
            positions: [B, N, 3] node positions (for kNN + distances)

        Returns:
            [B, N, hidden_dim] updated node features
        """
        B, N, D = h.shape
        k = min(self.k_neighbors, N - 1)

        # Compute kNN edges
        edge_idx, edge_dist, edge_mask = compute_knn_edges(
            positions, k, self.cutoff
        )

        # RBF encode distances: [B, N, k, n_rbf]
        rbf_features = self.rbf(edge_dist)

        # Gather neighbor features: [B, N, k, D]
        # edge_idx: [B, N, k] -> expand for gathering
        idx_expanded = edge_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        h_j = torch.gather(
            h.unsqueeze(2).expand(-1, -1, k, -1),
            dim=1,
            index=idx_expanded.long(),
        )
        # Simpler approach: use index_select per batch
        h_neighbors = []
        for b in range(B):
            # edge_idx[b]: [N, k]
            h_nb = h[b][edge_idx[b].long()]  # [N, k, D]
            h_neighbors.append(h_nb)
        h_j = torch.stack(h_neighbors, dim=0)  # [B, N, k, D]

        # Source node features repeated: [B, N, k, D]
        h_i = h.unsqueeze(2).expand(-1, -1, k, -1)

        # Edge features: [B, N, k, 2*D + n_rbf]
        edge_features = torch.cat([h_i, h_j, rbf_features], dim=-1)

        # Edge MLP -> messages: [B, N, k, D]
        messages = self.edge_mlp(edge_features)

        # Mask invalid edges
        messages = messages * edge_mask.unsqueeze(-1).float()

        # Aggregate: sum over neighbors
        agg_msg = messages.sum(dim=2)  # [B, N, D]

        # Node update
        node_input = torch.cat([h, agg_msg], dim=-1)  # [B, N, 2*D]
        update = self.node_mlp(node_input)

        return self.norm(h + update)


if __name__ == '__main__':
    print("Testing Local MPNN Module...")

    B, N, D = 4, 20, 128
    h = torch.randn(B, N, D)
    pos = torch.randn(B, N, 3) * 3.0

    # Test RBF
    rbf = RadialBasisFunction(n_rbf=20, cutoff=5.0)
    dists = torch.randn(B, N, 10).abs()
    rbf_out = rbf(dists)
    print(f"  RBF: input={dists.shape}, output={rbf_out.shape}")

    # Test kNN
    edge_idx, edge_dist, edge_mask = compute_knn_edges(pos, k=10, cutoff=5.0)
    print(f"  kNN: idx={edge_idx.shape}, dist={edge_dist.shape}, mask_pct={edge_mask.float().mean():.2f}")

    # Test MPNN layer
    mpnn = LocalMPNNLayer(hidden_dim=D, k_neighbors=10, cutoff=5.0)
    h_out = mpnn(h, pos)
    print(f"  MPNN: input={h.shape}, output={h_out.shape}")

    # Test gradient flow
    loss = h_out.sum()
    loss.backward()
    print(f"  Gradient flow: OK")

    # Test with different N
    for n in [5, 50, 100]:
        h_test = torch.randn(2, n, D)
        pos_test = torch.randn(2, n, 3) * 3.0
        h_out_test = mpnn(h_test, pos_test)
        print(f"  N={n}: output={h_out_test.shape}")

    print("\nSUCCESS: Local MPNN module working!")
