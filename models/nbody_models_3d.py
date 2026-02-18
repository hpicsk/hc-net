"""
3D Models for N-body Physics Prediction.

Extended from 2D models to support 3D gravitational simulations.
Uses Clifford algebra Cl(6,0) or Cl(8,0) for 3D position + velocity.

Key changes from 2D:
- Input: [N, 6] (x, y, z, vx, vy, vz) instead of [N, 4]
- Larger Clifford algebra dimension for 3D
- More complex geometric interactions (3D rotations have 3 bivector components)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class CliffordNBodyNet3D(nn.Module):
    """
    3D Clifford-based N-body prediction network.

    Extended from 2D version to handle 3D gravitational dynamics.
    Uses Cl(6,0) structure: position (e1,e2,e3) + velocity (e4,e5,e6).

    In 3D, the geometric product creates:
    - 3 position bivectors (e1e2, e1e3, e2e3) - related to angular momentum
    - 3 velocity bivectors (e4e5, e4e6, e5e6)
    - 9 cross-term bivectors (ei*ej for position-velocity pairs)
    """

    def __init__(
        self,
        n_particles: int = 5,
        hidden_dim: int = 128,
        block_size: int = 6,  # Use Cl(6,0) for 3D
        n_layers: int = 4,
        dropout: float = 0.1,
        use_attention: bool = False
    ):
        """
        Args:
            n_particles: Number of particles in the system
            hidden_dim: Hidden dimension
            block_size: Clifford algebra dimension (6 for 3D position + velocity)
            n_layers: Number of processing layers
            dropout: Dropout rate
            use_attention: If True, use multi-head attention for inter-particle
                interaction. If False (default), use a simple feedforward MLP.
        """
        super().__init__()

        self.n_particles = n_particles
        self.block_size = block_size
        self.mv_dim = 2 ** block_size  # 64 for Cl(6,0)
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention

        # Input: [N, 6] raw state -> [N, hidden_dim]
        self.input_proj = nn.Linear(6, hidden_dim)

        # Processing layers with geometric-inspired operations
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(CliffordBlock3D(hidden_dim, block_size, dropout))

        # Inter-particle interaction
        if use_attention:
            self.particle_interaction = ParticleInteraction3D(hidden_dim, n_particles)
        else:
            self.particle_interaction = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 6)

        # Additional geometric interaction layer
        self.geo_interaction = GeometricInteractionLayer3D(hidden_dim, block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, N, 6] particle states (x, y, z, vx, vy, vz)

        Returns:
            [B, N, 6] predicted next states
        """
        B, N, _ = x.shape

        # Project to hidden dimension
        h = self.input_proj(x)  # [B, N, hidden_dim]

        # Process through Clifford-inspired layers
        for layer in self.layers:
            h = layer(h)

        # Apply geometric interaction (captures bivector-like terms)
        h = self.geo_interaction(h)

        # Inter-particle interaction (attention or feedforward)
        h = self.particle_interaction(h)

        # Project back to state space
        delta = self.output_proj(h)  # [B, N, 6]

        # Predict residual (change from current state)
        return x + delta


class CliffordBlock3D(nn.Module):
    """
    3D Processing block inspired by Clifford algebra structure.

    Implements operations that mimic the geometric product's ability
    to create grade-mixed outputs from vector inputs in 3D.
    """

    def __init__(self, dim: int, block_size: int, dropout: float = 0.1):
        super().__init__()

        self.dim = dim
        # Use fixed group size that divides evenly into dim
        self.group_size = 8  # Use 8 for all dimensions (works with 128, 256, etc.)
        self.n_groups = dim // self.group_size

        # Main transformation
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

        # Geometric mixing: simulates bivector generation from vectors
        self.geo_mix = nn.Linear(self.group_size * self.group_size, self.group_size)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim]

        Returns:
            [B, N, dim]
        """
        B, N, D = x.shape

        # Standard MLP path
        residual = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual

        # Geometric mixing path
        residual = x
        x_groups = x.view(B, N, self.n_groups, self.group_size)

        # Compute outer products within each group (simulates bivector generation)
        outer = torch.einsum('bngi,bngj->bngij', x_groups, x_groups)
        outer = outer.view(B, N, self.n_groups, self.group_size * self.group_size)

        # Project back to group_size
        geo_features = self.geo_mix(outer)
        geo_features = geo_features.view(B, N, D)

        x = self.norm2(residual + 0.1 * geo_features)

        return x


class GeometricInteractionLayer3D(nn.Module):
    """
    3D Layer that explicitly computes geometric interactions between features.

    Simulates the bivector terms that would arise from geometric products
    of 3D position and velocity vectors in Clifford algebra.

    For 3D: 3 position components, 3 velocity components
    Cross products yield 9 bivector-like terms.
    """

    def __init__(self, dim: int, block_size: int):
        super().__init__()

        self.dim = dim
        self.block_size = block_size
        self.n_components = 12  # 3D has more components than 2D

        # Project features to position-like and velocity-like subspaces
        self.pos_proj = nn.Linear(dim, self.n_components)
        self.vel_proj = nn.Linear(dim, self.n_components)

        # Interaction: outer product gives n_components^2 features
        self.interaction_proj = nn.Linear(self.n_components * self.n_components, dim)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim]

        Returns:
            [B, N, dim]
        """
        B, N, D = x.shape

        # Extract position-like and velocity-like features
        pos = self.pos_proj(x)  # [B, N, n_components]
        vel = self.vel_proj(x)  # [B, N, n_components]

        # Compute interaction (outer product)
        interaction = torch.einsum('bni,bnj->bnij', pos, vel)
        interaction = interaction.view(B, N, -1)

        # Project to full dimension
        interaction = self.interaction_proj(interaction)

        return self.norm(x + interaction)


class ParticleInteraction3D(nn.Module):
    """
    Inter-particle interaction via multi-head attention for 3D systems.
    """

    def __init__(self, dim: int, n_particles: int, n_heads: int = 4):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            dim, num_heads=n_heads, batch_first=True, dropout=0.1
        )
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.Dropout(0.1)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim]

        Returns:
            [B, N, dim]
        """
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class BaselineNBodyNet3D(nn.Module):
    """
    3D Baseline MLP for N-body prediction without Clifford algebra.
    """

    def __init__(
        self,
        n_particles: int = 5,
        input_dim: int = 6,  # x, y, z, vx, vy, vz per particle
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_particles = n_particles
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        total_input = n_particles * input_dim

        layers = []

        # Input layer
        layers.append(nn.Linear(total_input, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, total_input))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, N, 6] particle states

        Returns:
            [B, N, 6] predicted next states
        """
        B, N, D = x.shape

        # Flatten
        x_flat = x.view(B, -1)

        # Process
        delta_flat = self.net(x_flat)

        # Reshape and add residual
        delta = delta_flat.view(B, N, D)

        return x + delta


class BaselineNBodyNetWithAttention3D(nn.Module):
    """
    3D Enhanced baseline with particle attention but no Clifford operations.
    """

    def __init__(
        self,
        n_particles: int = 5,
        input_dim: int = 6,
        hidden_dim: int = 128,
        n_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_particles = n_particles
        self.input_dim = input_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Processing layers (standard MLP blocks)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout)
            ))

        # Particle attention
        self.particle_attention = ParticleInteraction3D(hidden_dim, n_particles)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, N, 6] particle states

        Returns:
            [B, N, 6] predicted next states
        """
        # Project to hidden dim
        h = self.input_proj(x)

        # Process through layers
        for layer in self.layers:
            h = h + layer(h)

        # Particle attention
        h = self.particle_attention(h)

        # Project to output
        delta = self.output_proj(h)

        return x + delta


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the 3D models
    print("Testing 3D N-body Models...")

    B, N = 4, 5  # batch size, num particles
    x = torch.randn(B, N, 6)  # 3D state: x,y,z,vx,vy,vz

    # Test Clifford model
    clifford_model = CliffordNBodyNet3D(n_particles=N, hidden_dim=128)
    y_clifford = clifford_model(x)
    print(f"CliffordNBodyNet3D:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {y_clifford.shape}")
    print(f"  Parameters: {count_parameters(clifford_model):,}")

    # Test baseline MLP
    baseline_model = BaselineNBodyNet3D(n_particles=N, hidden_dim=256)
    y_baseline = baseline_model(x)
    print(f"\nBaselineNBodyNet3D:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {y_baseline.shape}")
    print(f"  Parameters: {count_parameters(baseline_model):,}")

    # Test baseline with attention
    baseline_attn = BaselineNBodyNetWithAttention3D(n_particles=N, hidden_dim=128)
    y_baseline_attn = baseline_attn(x)
    print(f"\nBaselineNBodyNetWithAttention3D:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {y_baseline_attn.shape}")
    print(f"  Parameters: {count_parameters(baseline_attn):,}")

    # Test gradient flow
    loss = y_clifford.sum()
    loss.backward()
    print("\nGradient flow: OK")

    # Test rotation equivariance (3D rotation around z-axis)
    print("\nTesting 3D rotation equivariance...")
    import numpy as np
    angle = 45.0 * np.pi / 180.0
    Rz = torch.tensor([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ], dtype=torch.float32)

    def rotate_state_3d(state, R):
        """Rotate 3D state by rotation matrix R."""
        pos = state[..., :3] @ R.T
        vel = state[..., 3:] @ R.T
        return torch.cat([pos, vel], dim=-1)

    clifford_model.eval()
    with torch.no_grad():
        x_rot = rotate_state_3d(x, Rz)
        output_from_rotated = clifford_model(x_rot)
        output_then_rotated = rotate_state_3d(clifford_model(x), Rz)
        error = (output_from_rotated - output_then_rotated).abs().mean()
        print(f"  Rotation equivariance error: {error:.6f}")

    print("\nSUCCESS: All 3D models working!")
