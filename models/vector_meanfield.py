"""
Vector Mean-Field Network - Strawman Baseline for HC-Net.

This model uses the SAME architecture as HC-Net but with vector-only features
(no bivector/geometric product operations). This demonstrates the "vector
averaging collapse" problem: when particles have zero net momentum (e.g.,
spinning system), the mean-field state contains no useful information.

Key differences from HC-Net:
1. NO geometric product / outer product operations (no bivector generation)
2. Only scalar + vector features (no grade-2 bivector components)
3. Same O(N) mean-field aggregation

Hypothesis: This baseline will FAIL on rotating systems because:
- Average velocity sum(v_i) -> 0 for symmetric rotations
- Without bivectors, mean-field cannot encode angular momentum
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class VectorMeanFieldNet(nn.Module):
    """
    Vector-only mean-field network (no Clifford/bivector operations).

    This is a strawman baseline showing that vector averaging alone
    cannot capture rotational dynamics in symmetric spinning systems.

    Architecture mirrors HC-Net but removes:
    - Geometric mixing (outer product -> bivector generation)
    - GeometricInteractionLayer (position-velocity outer products)

    Keeps:
    - Same hidden dimension
    - Same number of layers
    - Same mean-field aggregation (global average pooling)
    - Same residual connections
    """

    def __init__(
        self,
        n_particles: int = 5,
        hidden_dim: int = 128,
        n_layers: int = 4,
        dropout: float = 0.1,
        use_mean_field: bool = True
    ):
        """
        Args:
            n_particles: Number of particles in the system
            hidden_dim: Hidden dimension
            n_layers: Number of processing layers
            dropout: Dropout rate
            use_mean_field: If True (default), use mean-field global aggregation
        """
        super().__init__()

        self.n_particles = n_particles
        self.hidden_dim = hidden_dim
        self.use_mean_field = use_mean_field

        # Input projection: [N, 4] -> [N, hidden_dim]
        self.input_proj = nn.Linear(4, hidden_dim)

        # Processing layers (standard MLP blocks - NO geometric operations)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(VectorBlock(hidden_dim, dropout))

        # Mean-field aggregation (same as HC-Net)
        if use_mean_field:
            self.mean_field = VectorMeanFieldLayer(hidden_dim, n_components=8)

        # Simple feedforward interaction (NO attention, NO geometric ops)
        self.particle_interaction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, N, 4] particle states (x, y, vx, vy)

        Returns:
            [B, N, 4] predicted next states
        """
        B, N, _ = x.shape

        # Project to hidden dimension
        h = self.input_proj(x)  # [B, N, hidden_dim]

        # Process through vector blocks (NO geometric mixing)
        for layer in self.layers:
            h = layer(h)

        # Mean-field aggregation: O(N) inter-particle communication
        if self.use_mean_field:
            h = self.mean_field(h)

        # Inter-particle interaction (simple feedforward)
        h = self.particle_interaction(h)

        # Project back to state space
        delta = self.output_proj(h)  # [B, N, 4]

        # Predict residual
        return x + delta


class VectorBlock(nn.Module):
    """
    Standard MLP block WITHOUT geometric mixing.

    Compare to CliffordBlock which has outer product operations.
    This block is purely scalar/vector - no bivector generation.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()

        self.dim = dim

        # Standard MLP
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim]

        Returns:
            [B, N, dim]
        """
        # Standard MLP with residual
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual

        # NOTE: NO geometric mixing path here
        # In CliffordBlock, there would be outer products creating bivector-like features

        return x


class VectorMeanFieldLayer(nn.Module):
    """
    Vector-only mean-field aggregation layer.

    Same structure as MeanFieldAggregationLayer but explicitly
    operates on vector (not multivector) features.

    The key limitation: averaging vectors loses rotational information
    because sum(v_i) = 0 for symmetric rotating systems.
    """

    def __init__(self, dim: int, n_components: int = 8, scale: float = 0.1):
        """
        Args:
            dim: Hidden dimension
            n_components: Dimension of interaction space
            scale: Weight of mean-field contribution
        """
        super().__init__()
        self.dim = dim
        self.n_components = n_components
        self.scale = scale

        # Project particle features to interaction space
        self.particle_proj = nn.Linear(dim, n_components)
        # Project mean-field to interaction space
        self.meanfield_proj = nn.Linear(dim, n_components)
        # Project outer product back to hidden dim
        self.interaction_proj = nn.Linear(n_components * n_components, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim] particle features

        Returns:
            [B, N, dim] updated features with mean-field influence
        """
        B, N, D = x.shape

        # 1. Global mean-field: just averaging (NO bivector components)
        # This is where information is LOST for rotating systems:
        # If particles are spinning, their velocities cancel out!
        mean_field = x.mean(dim=1, keepdim=True)  # [B, 1, dim]

        # 2. Project to interaction space
        particle_feat = self.particle_proj(x)             # [B, N, n_comp]
        meanfield_feat = self.meanfield_proj(mean_field)  # [B, 1, n_comp]

        # 3. Outer product interaction
        meanfield_feat = meanfield_feat.expand(-1, N, -1)  # [B, N, n_comp]
        interaction = torch.einsum('bni,bnj->bnij', particle_feat, meanfield_feat)
        interaction = interaction.view(B, N, -1)  # [B, N, n_comp^2]

        # 4. Project back and residual
        out = self.interaction_proj(interaction)
        return self.norm(x + self.scale * out)


class VectorMeanFieldNetNaive(nn.Module):
    """
    Even simpler baseline: just MLP + vector mean concatenation.

    This is the most naive vector-based mean-field approach.
    """

    def __init__(
        self,
        n_particles: int = 5,
        hidden_dim: int = 128,
        n_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_particles = n_particles
        self.hidden_dim = hidden_dim

        # Input: particle features + mean features
        self.input_proj = nn.Linear(4 + 4, hidden_dim)  # concat with mean

        # Processing layers
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

        # Output
        self.output_proj = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 4] particle states

        Returns:
            [B, N, 4] predicted next states
        """
        B, N, D = x.shape

        # Compute mean state (THIS LOSES ROTATION INFO!)
        mean_state = x.mean(dim=1, keepdim=True).expand(-1, N, -1)  # [B, N, 4]

        # Concatenate particle with mean
        h = torch.cat([x, mean_state], dim=-1)  # [B, N, 8]
        h = self.input_proj(h)  # [B, N, hidden_dim]

        # Process
        for layer in self.layers:
            h = h + layer(h)

        # Output
        delta = self.output_proj(h)
        return x + delta


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the models
    print("Testing Vector Mean-Field Models...")

    B, N = 4, 5  # batch size, num particles
    x = torch.randn(B, N, 4)

    # Test VectorMeanFieldNet
    model = VectorMeanFieldNet(n_particles=N, hidden_dim=128)
    y = model(x)
    print(f"VectorMeanFieldNet:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Parameters: {count_parameters(model):,}")

    # Test VectorMeanFieldNetNaive
    model_naive = VectorMeanFieldNetNaive(n_particles=N, hidden_dim=128)
    y_naive = model_naive(x)
    print(f"\nVectorMeanFieldNetNaive:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {y_naive.shape}")
    print(f"  Parameters: {count_parameters(model_naive):,}")

    # Test gradient flow
    loss = y.sum()
    loss.backward()
    print("\nGradient flow: OK")

    print("\nSUCCESS: Vector mean-field models working!")
