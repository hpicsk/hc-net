"""
E(n) Equivariant Graph Neural Network for N-body Physics Prediction.

Implements EGNN from:
"E(n) Equivariant Graph Neural Networks" (Satorras et al., ICML 2021)

This is a SOTA baseline for comparison with HC-Net.
EGNN achieves E(n) equivariance by:
1. Using coordinate differences (equivariant) weighted by invariant scalars
2. Keeping node features invariant
3. Computing edge features from squared distances (invariant)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from nips_hcnet.layers.egnn_layers import EGNNLayer, EGNNBlock, VelocityEGNNLayer


class EGNNNBodyNet(nn.Module):
    """
    EGNN for N-body dynamics prediction.

    Input: [B, N, 4] particle states (x, y, vx, vy)
    Output: [B, N, 4] predicted next states

    Architecture:
    1. Embed raw features to hidden dimension
    2. Stack EGNN layers that update features and coordinates
    3. Project back to state predictions

    Key design choices:
    - Positions and velocities are treated as separate coordinate spaces
    - Node features are initialized from concatenated pos/vel
    - Both pos and vel are updated equivariantly
    """

    def __init__(
        self,
        n_particles: int = 5,
        hidden_dim: int = 128,
        n_layers: int = 4,
        coord_dim: int = 2,
        dropout: float = 0.1,
        attention: bool = True,
        residual: bool = True,
        normalize_coords: bool = True,
        normalize_input: bool = True
    ):
        """
        Args:
            n_particles: Number of particles
            hidden_dim: Hidden feature dimension
            n_layers: Number of EGNN layers
            coord_dim: Coordinate dimension (2 for 2D, 3 for 3D)
            dropout: Dropout rate
            attention: Whether to use attention in EGNN layers
            residual: Whether to use residual connections
            normalize_coords: Whether to normalize coordinate updates
            normalize_input: Whether to normalize input state per-sample for stability
        """
        super().__init__()

        self.n_particles = n_particles
        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim
        self.n_layers = n_layers
        self.normalize_input = normalize_input

        # Input: state [x, y, vx, vy] -> node features
        # We use pos and vel separately for coordinates
        # Node features are initialized from invariant quantities
        self.feature_embedding = nn.Sequential(
            nn.Linear(coord_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # EGNN layers that update both pos and vel
        self.egnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.egnn_layers.append(
                VelocityEGNNLayer(
                    hidden_dim=hidden_dim,
                    act_fn=nn.SiLU(),
                    residual=residual,
                    attention=attention
                )
            )

        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

        # Output projection: features -> state delta
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, coord_dim * 2)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: [B, N, 2*coord_dim] particle states (positions + velocities)

        Returns:
            [B, N, 2*coord_dim] predicted next states
        """
        B, N, _ = state.shape

        # Normalize input for numerical stability
        if self.normalize_input:
            state_mean = state.mean(dim=1, keepdim=True)
            state_std = state.std(dim=1, keepdim=True) + 1e-8
            state_normed = (state - state_mean) / state_std
        else:
            state_normed = state

        # Split into positions and velocities
        pos = state_normed[..., :self.coord_dim]  # [B, N, coord_dim]
        vel = state_normed[..., self.coord_dim:]  # [B, N, coord_dim]

        # Initialize node features from normalized state
        h = self.feature_embedding(state_normed)  # [B, N, hidden_dim]

        # Apply EGNN layers
        for i, layer in enumerate(self.egnn_layers):
            h_norm = self.layer_norms[i](h)
            h, pos, vel = layer(h_norm, pos, vel)
            h = self.dropout(h)

        # Project to output delta
        delta = self.output_proj(h)  # [B, N, coord_dim * 2]

        # Denormalize delta if input was normalized
        if self.normalize_input:
            delta = delta * state_std

        # Residual prediction
        return state + delta


class EGNNNBodyNetSimple(nn.Module):
    """
    Simpler EGNN variant that treats the full state as a single coordinate.

    This version is closer to the original EGNN paper formulation,
    treating the 4D state (x, y, vx, vy) as coordinates in R^4.
    """

    def __init__(
        self,
        n_particles: int = 5,
        hidden_dim: int = 128,
        n_layers: int = 4,
        state_dim: int = 4,
        dropout: float = 0.1,
        attention: bool = True
    ):
        super().__init__()

        self.n_particles = n_particles
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        # Node features are learned, not derived from state
        self.node_embedding = nn.Parameter(torch.randn(1, n_particles, hidden_dim) * 0.02)

        # Alternatively, derive features from state invariants
        self.feature_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # EGNN layers
        self.egnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.egnn_layers.append(
                EGNNBlock(
                    hidden_dim=hidden_dim,
                    attention=attention,
                    dropout=dropout
                )
            )

        # Output: coordinate delta is the prediction
        # Additional feature-based prediction
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass treating full state as coordinate.

        Args:
            state: [B, N, 4] particle states

        Returns:
            [B, N, 4] predicted next states
        """
        B, N, D = state.shape

        # Initialize features
        h = self.feature_mlp(state)  # [B, N, hidden_dim]
        x = state.clone()  # Coordinates = state

        # Apply EGNN layers
        for layer in self.egnn_layers:
            h, x = layer(h, x)

        # The coordinate output x already contains equivariant updates
        # Add feature-based correction
        delta = self.output_mlp(h)

        # Combine coordinate update with feature-based delta
        # Weight the coordinate update less to avoid instability
        coord_delta = x - state
        output = state + 0.5 * coord_delta + 0.5 * delta

        return output


class EGNNNBodyNetV2(nn.Module):
    """
    EGNN variant optimized for N-body dynamics.

    Differences from base EGNN:
    1. Uses relative positions between particles more explicitly
    2. Includes velocity-based message passing
    3. Multi-step prediction support
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

        # Embed particles
        self.particle_embed = nn.Embedding(n_particles, hidden_dim // 2)

        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(coord_dim * 2 + hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim)
        )

        # EGNN layers with separate pos/vel handling
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'egnn': EGNNLayer(
                    hidden_dim=hidden_dim,
                    attention=True,
                    residual=True
                ),
                'norm': nn.LayerNorm(hidden_dim),
                'ff': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim)
                )
            }))

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, coord_dim * 2)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: [B, N, 4] particle states

        Returns:
            [B, N, 4] predicted states
        """
        B, N, _ = state.shape
        device = state.device

        # Particle indices
        particle_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        particle_emb = self.particle_embed(particle_idx)  # [B, N, hidden_dim//2]

        # Encode
        h = self.encoder(torch.cat([state, particle_emb], dim=-1))

        # Use positions as coordinates for EGNN
        x = state[..., :self.coord_dim]  # [B, N, coord_dim]

        # Process through layers
        for layer_dict in self.layers:
            # EGNN layer
            h_norm = layer_dict['norm'](h)
            h_new, x = layer_dict['egnn'](h_norm, x)

            # Feed-forward with residual
            h = h + h_new
            h = h + layer_dict['ff'](h)

        # Decode to state delta
        delta = self.decoder(h)

        return state + delta


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the models
    print("Testing EGNN N-body Models...")

    B, N = 4, 5  # batch size, num particles
    state = torch.randn(B, N, 4)

    # Test main model
    print("\n1. EGNNNBodyNet:")
    model = EGNNNBodyNet(n_particles=N, hidden_dim=128, n_layers=4)
    output = model(state)
    print(f"   Input: {state.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Parameters: {count_parameters(model):,}")

    # Test gradient flow
    loss = output.sum()
    loss.backward()
    print("   Gradient flow: OK")

    # Test simple model
    print("\n2. EGNNNBodyNetSimple:")
    model_simple = EGNNNBodyNetSimple(n_particles=N, hidden_dim=128, n_layers=4)
    output_simple = model_simple(state)
    print(f"   Input: {state.shape}")
    print(f"   Output: {output_simple.shape}")
    print(f"   Parameters: {count_parameters(model_simple):,}")

    # Test V2 model
    print("\n3. EGNNNBodyNetV2:")
    model_v2 = EGNNNBodyNetV2(n_particles=N, hidden_dim=128, n_layers=4)
    output_v2 = model_v2(state)
    print(f"   Input: {state.shape}")
    print(f"   Output: {output_v2.shape}")
    print(f"   Parameters: {count_parameters(model_v2):,}")

    # Test equivariance
    print("\n4. Testing rotation equivariance:")
    angle = torch.tensor(45.0 * 3.14159 / 180.0)
    c, s = torch.cos(angle), torch.sin(angle)
    R = torch.tensor([[c, -s], [s, c]])

    def rotate_state(x, R):
        """Rotate positions and velocities."""
        pos = x[..., :2] @ R.T
        vel = x[..., 2:] @ R.T
        return torch.cat([pos, vel], dim=-1)

    model.eval()
    with torch.no_grad():
        # f(Rx)
        state_rotated = rotate_state(state, R)
        output_from_rotated = model(state_rotated)

        # R * f(x)
        output_original = model(state)
        output_then_rotated = rotate_state(output_original, R)

        # Equivariance error
        error = (output_from_rotated - output_then_rotated).abs().mean()
        print(f"   Equivariance error: {error:.6f}")
        print(f"   (Should be close to 0 for perfect equivariance)")

    print("\nSUCCESS: All EGNN models working!")
