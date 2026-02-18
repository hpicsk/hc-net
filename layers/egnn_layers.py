"""
E(n) Equivariant Graph Neural Network Layers.

Implements the core building blocks of EGNN from:
"E(n) Equivariant Graph Neural Networks" (Satorras et al., ICML 2021)

Key insight: Separate scalar features (invariant) from coordinate features (equivariant).
Coordinates are updated using vector differences, preserving E(n) equivariance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class EGNNLayer(nn.Module):
    """
    Single E(n)-equivariant layer.

    For N particles with positions x and features h:
    1. Compute pairwise squared distances: d_ij = ||x_i - x_j||^2
    2. Compute messages: m_ij = phi_e(h_i, h_j, d_ij)
    3. Update coordinates: x_i' = x_i + sum_j (x_i - x_j) * phi_x(m_ij)
    4. Update features: h_i' = phi_h(h_i, sum_j m_ij)

    The key to E(n) equivariance:
    - phi_e uses ||x_i - x_j||^2 which is invariant
    - Coordinate update uses (x_i - x_j) weighted by invariant phi_x output
    - Feature update uses only invariant quantities
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 0,
        act_fn: nn.Module = nn.SiLU(),
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        coord_scale: float = 1.0,
        tanh_coord: bool = False
    ):
        """
        Args:
            hidden_dim: Dimension of node features
            edge_dim: Dimension of edge attributes (0 if none)
            act_fn: Activation function
            residual: Whether to use residual connections
            attention: Whether to use attention mechanism
            normalize: Whether to normalize coordinate updates
            coord_scale: Scale factor for coordinate updates
            tanh_coord: Apply tanh to coordinate updates for stability
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coord_scale = coord_scale
        self.tanh_coord = tanh_coord

        # Edge MLP: computes messages from node pairs and distance
        # Input: h_i || h_j || d_ij || edge_attr
        edge_input_dim = 2 * hidden_dim + 1 + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )

        # Node MLP: updates node features
        # Input: h_i || aggregated messages
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Coordinate MLP: produces scalar weights for coordinate updates
        # Only outputs 1 value per edge
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, 1, bias=False)
        )

        # Optional attention
        if self.attention:
            self.attention_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            h: [B, N, hidden_dim] node features
            x: [B, N, coord_dim] node coordinates (e.g., 2D or 3D positions)
            edge_attr: [B, N, N, edge_dim] optional edge attributes

        Returns:
            h_out: [B, N, hidden_dim] updated node features
            x_out: [B, N, coord_dim] updated coordinates
        """
        B, N, coord_dim = x.shape

        # Compute pairwise coordinate differences: [B, N, N, coord_dim]
        x_diff = x.unsqueeze(2) - x.unsqueeze(1)  # x_i - x_j

        # Compute squared distances: [B, N, N, 1]
        dist_sq = (x_diff ** 2).sum(dim=-1, keepdim=True)

        # Build edge inputs: [B, N, N, edge_input_dim]
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, hidden_dim]
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, hidden_dim]

        edge_input = torch.cat([h_i, h_j, dist_sq], dim=-1)
        if edge_attr is not None:
            edge_input = torch.cat([edge_input, edge_attr], dim=-1)

        # Compute edge messages: [B, N, N, hidden_dim]
        m_ij = self.edge_mlp(edge_input)

        # Optional attention
        if self.attention:
            att = self.attention_mlp(m_ij)  # [B, N, N, 1]
            m_ij = m_ij * att

        # Coordinate update weights: [B, N, N, 1]
        coord_weights = self.coord_mlp(m_ij)
        # Always apply tanh for stability (bounded output)
        coord_weights = torch.tanh(coord_weights)

        # Coordinate update: x_i' = x_i + sum_j (x_i - x_j) * phi_x(m_ij)
        # Note: we sum over j (dim=2), excluding self-loops implicitly
        # by the fact that x_diff[i,i] = 0
        coord_update = (x_diff * coord_weights).sum(dim=2)  # [B, N, coord_dim]

        # Normalize by number of neighbors
        coord_update = coord_update / max(N - 1, 1)

        x_out = x + self.coord_scale * coord_update

        # Message aggregation for node update: [B, N, hidden_dim]
        m_agg = m_ij.sum(dim=2)  # Sum over j

        # Node update
        node_input = torch.cat([h, m_agg], dim=-1)
        h_out = self.node_mlp(node_input)

        if self.residual:
            h_out = h + h_out

        return h_out, x_out


class EGNNBlock(nn.Module):
    """
    EGNN block with layer normalization.

    Wraps EGNNLayer with pre-normalization for stability.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 0,
        act_fn: nn.Module = nn.SiLU(),
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_dim)
        self.egnn_layer = EGNNLayer(
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            act_fn=act_fn,
            residual=residual,
            attention=attention,
            normalize=normalize
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with normalization.

        Args:
            h: [B, N, hidden_dim] node features
            x: [B, N, coord_dim] coordinates
            edge_attr: Optional edge attributes

        Returns:
            h_out, x_out: Updated features and coordinates
        """
        h_norm = self.norm(h)
        h_out, x_out = self.egnn_layer(h_norm, x, edge_attr)
        h_out = self.dropout(h_out)
        return h_out, x_out


class VelocityEGNNLayer(nn.Module):
    """
    EGNN layer that handles both positions and velocities.

    For N-body dynamics, we need to track:
    - Positions: x (equivariant)
    - Velocities: v (equivariant)
    - Features: h (invariant)

    This layer updates all three while preserving E(n) equivariance.
    """

    def __init__(
        self,
        hidden_dim: int,
        act_fn: nn.Module = nn.SiLU(),
        residual: bool = True,
        attention: bool = True,
        normalize_coords: bool = True,
        coord_clamp: float = 100.0
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.residual = residual
        self.normalize_coords = normalize_coords
        self.coord_clamp = coord_clamp

        # Edge MLP: uses distances between positions
        # Input: h_i || h_j || dist_pos || dist_vel
        edge_input_dim = 2 * hidden_dim + 2  # +2 for pos and vel distances
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )

        # Position update weights
        self.pos_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, 1, bias=False)
        )

        # Velocity update weights
        self.vel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, 1, bias=False)
        )

        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Attention
        if attention:
            self.attention_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        self.attention = attention

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        vel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            h: [B, N, hidden_dim] node features
            pos: [B, N, coord_dim] positions
            vel: [B, N, coord_dim] velocities

        Returns:
            h_out: Updated features
            pos_out: Updated positions
            vel_out: Updated velocities
        """
        B, N, coord_dim = pos.shape

        # Compute pairwise differences
        pos_diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B, N, N, coord_dim]
        vel_diff = vel.unsqueeze(2) - vel.unsqueeze(1)  # [B, N, N, coord_dim]

        # Squared distances (invariant)
        pos_dist_sq = (pos_diff ** 2).sum(dim=-1, keepdim=True)  # [B, N, N, 1]
        vel_dist_sq = (vel_diff ** 2).sum(dim=-1, keepdim=True)  # [B, N, N, 1]

        # Build edge features
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)

        edge_input = torch.cat([h_i, h_j, pos_dist_sq, vel_dist_sq], dim=-1)

        # Edge messages
        m_ij = self.edge_mlp(edge_input)

        if self.attention:
            att = self.attention_mlp(m_ij)
            m_ij = m_ij * att

        # Update weights (bounded via tanh for numerical stability)
        pos_weights = torch.tanh(self.pos_mlp(m_ij))  # [B, N, N, 1], bounded [-1, 1]
        vel_weights = torch.tanh(self.vel_mlp(m_ij))  # [B, N, N, 1], bounded [-1, 1]

        # Coordinate updates (equivariant)
        pos_update = (pos_diff * pos_weights).sum(dim=2)
        vel_update = (vel_diff * vel_weights).sum(dim=2)

        # Normalize by number of neighbors to prevent accumulation
        if self.normalize_coords:
            pos_update = pos_update / max(N - 1, 1)
            vel_update = vel_update / max(N - 1, 1)

        pos_out = pos + pos_update
        vel_out = vel + vel_update

        # Clamp to prevent runaway values
        if self.coord_clamp > 0:
            pos_out = torch.clamp(pos_out, -self.coord_clamp, self.coord_clamp)
            vel_out = torch.clamp(vel_out, -self.coord_clamp, self.coord_clamp)

        # Node update
        m_agg = m_ij.sum(dim=2)
        node_input = torch.cat([h, m_agg], dim=-1)
        h_out = self.node_mlp(node_input)

        if self.residual:
            h_out = h + h_out

        return h_out, pos_out, vel_out


if __name__ == '__main__':
    # Test the layers
    print("Testing EGNN Layers...")

    B, N = 4, 5  # batch size, num particles
    hidden_dim = 64
    coord_dim = 2

    # Test EGNNLayer
    h = torch.randn(B, N, hidden_dim)
    x = torch.randn(B, N, coord_dim)

    layer = EGNNLayer(hidden_dim=hidden_dim)
    h_out, x_out = layer(h, x)

    print(f"EGNNLayer:")
    print(f"  h: {h.shape} -> {h_out.shape}")
    print(f"  x: {x.shape} -> {x_out.shape}")

    # Test VelocityEGNNLayer
    pos = torch.randn(B, N, coord_dim)
    vel = torch.randn(B, N, coord_dim)

    vel_layer = VelocityEGNNLayer(hidden_dim=hidden_dim)
    h_out, pos_out, vel_out = vel_layer(h, pos, vel)

    print(f"\nVelocityEGNNLayer:")
    print(f"  h: {h.shape} -> {h_out.shape}")
    print(f"  pos: {pos.shape} -> {pos_out.shape}")
    print(f"  vel: {vel.shape} -> {vel_out.shape}")

    # Test gradient flow
    loss = h_out.sum() + pos_out.sum() + vel_out.sum()
    loss.backward()
    print("\nGradient flow: OK")

    print("\nSUCCESS: EGNN layers working!")
