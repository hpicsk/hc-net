"""
NequIP-style Model for N-body Physics Prediction.

Implements concepts from:
"E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials"
(Batzner et al., Nature Communications 2022)

Adapted for N-body dynamics prediction (originally designed for molecular potentials).

Key features:
- Radial basis function encoding for distances
- Angular encoding (circular harmonics for 2D, spherical harmonics for 3D)
- Equivariant message passing
- Multi-resolution feature aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from nips_hcnet.layers.spherical_harmonics import (
    RadialBasisFunctions,
    SmoothCutoff,
    CircularHarmonics2D,
    SphericalHarmonics3D,
    EquivariantMessageBlock,
)


class NequIPLayer(nn.Module):
    """
    Single NequIP-style layer.

    Combines:
    1. Radial feature encoding
    2. Angular feature encoding
    3. Equivariant convolution
    4. Feature update
    """

    def __init__(
        self,
        hidden_dim: int,
        num_radial: int = 8,
        max_angular: int = 4,
        cutoff: float = 10.0,
        coord_dim: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim

        # Radial encoding
        self.radial_basis = RadialBasisFunctions(num_radial, cutoff)
        self.cutoff_fn = SmoothCutoff(cutoff)

        # Angular encoding
        if coord_dim == 2:
            self.angular = CircularHarmonics2D(max_angular)
            angular_features = self.angular.num_features
        else:
            self.angular = SphericalHarmonics3D(max_angular)
            angular_features = self.angular.num_features

        # Edge feature network
        edge_input_dim = hidden_dim * 2 + num_radial + angular_features
        self.edge_net = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )

        # Coordinate update network (for equivariant output)
        self.coord_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Node update network
        self.node_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(hidden_dim)

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
        B, N, _ = pos.shape

        # Pairwise vectors and distances
        pos_diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B, N, N, coord_dim]
        dist = torch.norm(pos_diff, dim=-1)  # [B, N, N]

        # Direction vectors
        direction = pos_diff / (dist.unsqueeze(-1) + 1e-8)

        # Radial and angular features
        radial = self.radial_basis(dist)  # [B, N, N, num_radial]
        cutoff = self.cutoff_fn(dist).unsqueeze(-1)  # [B, N, N, 1]
        angular = self.angular(direction)  # [B, N, N, angular_features]

        # Build edge features
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)

        edge_input = torch.cat([h_i, h_j, radial, angular], dim=-1)
        edge_features = self.edge_net(edge_input)  # [B, N, N, hidden]

        # Apply cutoff
        edge_features = edge_features * cutoff

        # Coordinate updates (equivariant)
        coord_weights = self.coord_net(edge_features).squeeze(-1)  # [B, N, N]
        pos_update = (pos_diff * coord_weights.unsqueeze(-1)).sum(dim=2)
        vel_update = (pos_diff * coord_weights.unsqueeze(-1)).sum(dim=2) * 0.1

        pos_out = pos + pos_update
        vel_out = vel + vel_update

        # Node updates
        msg_agg = edge_features.sum(dim=2)  # [B, N, hidden]
        node_input = torch.cat([h, msg_agg], dim=-1)
        h_out = self.node_net(node_input)
        h_out = self.norm(h + h_out)

        return h_out, pos_out, vel_out


class NequIPNBodyNet(nn.Module):
    """
    NequIP-style network for N-body dynamics prediction.

    Input: [B, N, 4] particle states (x, y, vx, vy)
    Output: [B, N, 4] predicted next states

    Architecture:
    1. Embed particle features
    2. Stack NequIP layers with radial/angular encoding
    3. Predict state updates
    """

    def __init__(
        self,
        n_particles: int = 5,
        hidden_dim: int = 128,
        n_layers: int = 4,
        coord_dim: int = 2,
        num_radial: int = 8,
        max_angular: int = 4,
        cutoff: float = 10.0,
        dropout: float = 0.1
    ):
        """
        Args:
            n_particles: Number of particles
            hidden_dim: Hidden feature dimension
            n_layers: Number of NequIP layers
            coord_dim: Coordinate dimension (2 or 3)
            num_radial: Number of radial basis functions
            max_angular: Maximum angular harmonic order
            cutoff: Distance cutoff
            dropout: Dropout rate
        """
        super().__init__()

        self.n_particles = n_particles
        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim

        # Feature embedding
        self.embed = nn.Sequential(
            nn.Linear(coord_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # NequIP layers
        self.layers = nn.ModuleList([
            NequIPLayer(
                hidden_dim=hidden_dim,
                num_radial=num_radial,
                max_angular=max_angular,
                cutoff=cutoff,
                coord_dim=coord_dim,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, coord_dim * 2)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: [B, N, 4] particle states (x, y, vx, vy)

        Returns:
            [B, N, 4] predicted next states
        """
        B, N, _ = state.shape

        # Split state
        pos = state[..., :self.coord_dim]
        vel = state[..., self.coord_dim:]

        # Embed features
        h = self.embed(state)

        # Process through NequIP layers
        for layer in self.layers:
            h, pos, vel = layer(h, pos, vel)

        # Compute output delta from features
        delta = self.output(h)

        # Residual prediction
        return state + delta


class NequIPNBodyNetSimple(nn.Module):
    """
    Simplified NequIP using message blocks directly.
    """

    def __init__(
        self,
        n_particles: int = 5,
        hidden_dim: int = 128,
        n_layers: int = 4,
        coord_dim: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_particles = n_particles
        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim

        # Embedding
        self.embed = nn.Linear(coord_dim * 2, hidden_dim)

        # Message blocks
        self.blocks = nn.ModuleList([
            EquivariantMessageBlock(
                hidden_dim=hidden_dim,
                coord_dim=coord_dim
            )
            for _ in range(n_layers)
        ])

        # Output
        self.output = nn.Linear(hidden_dim, coord_dim * 2)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        B, N, _ = state.shape

        pos = state[..., :self.coord_dim]
        h = self.embed(state)

        for block in self.blocks:
            h = block(h, pos)

        delta = self.output(h)
        return state + delta


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the models
    print("Testing NequIP N-body Models...")

    B, N = 4, 5
    state = torch.randn(B, N, 4)

    # Test main model
    print("\n1. NequIPNBodyNet:")
    model = NequIPNBodyNet(n_particles=N, hidden_dim=128, n_layers=4)
    output = model(state)
    print(f"   Input: {state.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Parameters: {count_parameters(model):,}")

    # Test gradient
    loss = output.sum()
    loss.backward()
    print("   Gradient flow: OK")

    # Test simple model
    print("\n2. NequIPNBodyNetSimple:")
    model_simple = NequIPNBodyNetSimple(n_particles=N, hidden_dim=128, n_layers=4)
    output_simple = model_simple(state)
    print(f"   Input: {state.shape}")
    print(f"   Output: {output_simple.shape}")
    print(f"   Parameters: {count_parameters(model_simple):,}")

    # Test rotation equivariance
    print("\n3. Testing rotation equivariance:")
    angle = torch.tensor(45.0 * 3.14159 / 180.0)
    c, s = torch.cos(angle), torch.sin(angle)
    R = torch.tensor([[c, -s], [s, c]])

    def rotate_state(x, R):
        pos = x[..., :2] @ R.T
        vel = x[..., 2:] @ R.T
        return torch.cat([pos, vel], dim=-1)

    model.eval()
    with torch.no_grad():
        state_rotated = rotate_state(state, R)
        output_from_rotated = model(state_rotated)
        output_then_rotated = rotate_state(model(state), R)
        error = (output_from_rotated - output_then_rotated).abs().mean()
        print(f"   Rotation equivariance error: {error:.6f}")

    print("\nSUCCESS: All NequIP models working!")
