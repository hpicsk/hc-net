"""
Models for N-body Physics Prediction.

Compares Clifford-based models (leveraging geometric product for
particle interactions) against standard MLP baselines.

Key architectural choice: Use Clifford layers to process particle
states, leveraging geometric product for interactions between
position and velocity vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class CliffordNBodyNet(nn.Module):
    """
    Clifford-based N-body prediction network.

    Uses Clifford algebra embedding and geometric product to naturally
    capture interactions between position and velocity vectors.

    The key insight: in Clifford algebra, the geometric product of
    two vectors naturally creates bivector terms that encode their
    relationship (angle, relative magnitude). This is exactly what's
    needed for physics simulation where forces depend on relative
    positions and momenta.

    Architecture:
    1. Embed particle states (x, y, vx, vy) as Clifford vectors
    2. Process through Clifford linear layers with geometric product
    3. Inter-particle attention for N-body interactions
    4. Decode back to state predictions
    """

    def __init__(
        self,
        n_particles: int = 5,
        hidden_dim: int = 128,
        block_size: int = 4,  # Use Cl(4,0) for efficiency
        n_layers: int = 4,
        dropout: float = 0.1,
        use_attention: bool = False,
        use_mean_field: bool = True
    ):
        """
        Args:
            n_particles: Number of particles in the system
            hidden_dim: Hidden dimension (will be multiple of mv_dim)
            block_size: Clifford algebra dimension (4 for position + velocity)
            n_layers: Number of processing layers
            dropout: Dropout rate
            use_attention: If True, use multi-head attention for inter-particle
                interaction. If False (default), use a simple feedforward MLP.
                The simpler variant often performs better (see ablation study).
            use_mean_field: If True (default), use mean-field global aggregation
                for O(N) inter-particle communication. This enables particles to
                interact with the collective system state without O(N^2) pairwise
                message passing.
        """
        super().__init__()

        self.n_particles = n_particles
        self.block_size = block_size
        self.mv_dim = 2 ** block_size  # 16 for Cl(4,0)
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        self.use_mean_field = use_mean_field

        # Input: [N, 4] raw state -> [N, hidden_dim]
        # We process in Clifford-inspired way but without full algebra
        # to keep computation tractable

        # Input projection
        self.input_proj = nn.Linear(4, hidden_dim)

        # Processing layers with geometric-inspired operations
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(CliffordBlock(hidden_dim, block_size, dropout))

        # Mean-field global aggregation for O(N) inter-particle communication
        if use_mean_field:
            self.mean_field = MeanFieldAggregationLayer(hidden_dim, n_components=8)

        # Inter-particle interaction
        if use_attention:
            self.particle_interaction = ParticleInteraction(hidden_dim, n_particles)
        else:
            self.particle_interaction = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 4)

        # Additional geometric interaction layer
        self.geo_interaction = GeometricInteractionLayer(hidden_dim, block_size)

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

        # Process through Clifford-inspired layers
        for layer in self.layers:
            h = layer(h)

        # Mean-field aggregation: O(N) inter-particle communication
        # Each particle interacts with the global mean state
        if self.use_mean_field:
            h = self.mean_field(h)

        # Apply geometric interaction (captures bivector-like terms)
        h = self.geo_interaction(h)

        # Inter-particle interaction (attention or feedforward)
        h = self.particle_interaction(h)

        # Project back to state space
        delta = self.output_proj(h)  # [B, N, 4]

        # Predict residual (change from current state)
        return x + delta


class CliffordBlock(nn.Module):
    """
    Processing block inspired by Clifford algebra structure.

    Implements operations that mimic the geometric product's ability
    to create grade-mixed outputs from vector inputs.
    """

    def __init__(self, dim: int, block_size: int, dropout: float = 0.1):
        super().__init__()

        self.dim = dim
        self.block_size = block_size

        # Main transformation
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

        # Geometric mixing: simulates bivector generation from vectors
        # Split features into groups and compute pairwise products
        self.n_groups = dim // block_size
        self.geo_mix = nn.Linear(block_size * block_size, block_size)

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
        x_groups = x.view(B, N, self.n_groups, self.block_size)

        # Compute outer products within each group (simulates bivector generation)
        # This creates block_size^2 features from block_size features
        outer = torch.einsum('bngi,bngj->bngij', x_groups, x_groups)
        outer = outer.view(B, N, self.n_groups, self.block_size * self.block_size)

        # Project back to block_size
        geo_features = self.geo_mix(outer)
        geo_features = geo_features.view(B, N, D)

        x = self.norm2(residual + 0.1 * geo_features)  # Small contribution

        return x


class GeometricInteractionLayer(nn.Module):
    """
    Layer that explicitly computes geometric interactions between features.

    Simulates the bivector terms that would arise from geometric products
    of position and velocity vectors in Clifford algebra.
    """

    def __init__(self, dim: int, block_size: int):
        super().__init__()

        self.dim = dim
        self.block_size = block_size
        self.n_components = 8  # Fixed size for outer product

        # Position-velocity interaction
        # In Cl(4,0): e1*e3, e1*e4, e2*e3, e2*e4 are bivectors
        # We simulate this by computing products of corresponding feature groups

        # Project features to position-like and velocity-like subspaces
        self.pos_proj = nn.Linear(dim, self.n_components)
        self.vel_proj = nn.Linear(dim, self.n_components)

        # Interaction: outer product gives n_components^2 = 64 features
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
        interaction = interaction.view(B, N, -1)  # [B, N, 64]

        # Project to full dimension
        interaction = self.interaction_proj(interaction)

        return self.norm(x + interaction)


class MeanFieldAggregationLayer(nn.Module):
    """
    Global mean-field aggregation for O(N) inter-particle communication.

    Physics motivation: In gravitational systems, each particle is influenced
    by the collective mean-field potential, approximating sum of all interactions.
    This is analogous to mean-field theory in statistical physics.

    Complexity: O(N) - compute mean once, broadcast and interact per-particle.
    Compare to O(N²) for explicit pairwise message passing (EGNN, CGENN).

    Architecture:
    1. Compute global mean-field: mean(h) over all particles
    2. Project particle features and mean-field to interaction space
    3. Compute outer product: particle_i × mean_field
    4. Project back and add residually
    """

    def __init__(self, dim: int, n_components: int = 8, scale: float = 0.1):
        """
        Args:
            dim: Hidden dimension
            n_components: Dimension of interaction space (default 8)
            scale: Weight of mean-field contribution (default 0.1)
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

        # 1. Global mean-field: O(N) to compute
        mean_field = x.mean(dim=1, keepdim=True)  # [B, 1, dim]

        # 2. Project to interaction space
        particle_feat = self.particle_proj(x)            # [B, N, n_comp]
        meanfield_feat = self.meanfield_proj(mean_field)  # [B, 1, n_comp]

        # 3. Outer product interaction: O(N)
        # Broadcast mean-field to all particles then compute outer product
        meanfield_feat = meanfield_feat.expand(-1, N, -1)  # [B, N, n_comp]
        interaction = torch.einsum('bni,bnj->bnij', particle_feat, meanfield_feat)
        interaction = interaction.view(B, N, -1)  # [B, N, n_comp²]

        # 4. Project back and residual connection
        out = self.interaction_proj(interaction)
        return self.norm(x + self.scale * out)


class ParticleInteraction(nn.Module):
    """
    Inter-particle interaction via multi-head attention.

    Allows particles to exchange information about their relative
    positions and velocities.
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
        # Self-attention over particles
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)

        # Feed-forward
        x = self.norm2(x + self.ff(x))

        return x


class BaselineNBodyNet(nn.Module):
    """
    Baseline MLP for N-body prediction without Clifford algebra.

    Standard approach: concatenate all particle states and process
    through MLP layers.
    """

    def __init__(
        self,
        n_particles: int = 5,
        input_dim: int = 4,  # x, y, vx, vy per particle
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_particles = n_particles
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Flatten all particles
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
            x: [B, N, 4] particle states

        Returns:
            [B, N, 4] predicted next states
        """
        B, N, D = x.shape

        # Flatten
        x_flat = x.view(B, -1)

        # Process
        delta_flat = self.net(x_flat)

        # Reshape and add residual
        delta = delta_flat.view(B, N, D)

        return x + delta


class BaselineNBodyNetWithAttention(nn.Module):
    """
    Enhanced baseline with particle attention but no Clifford operations.

    This is a fairer comparison as it also uses attention for
    inter-particle interactions.
    """

    def __init__(
        self,
        n_particles: int = 5,
        input_dim: int = 4,
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
        self.particle_attention = ParticleInteraction(hidden_dim, n_particles)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, N, 4] particle states

        Returns:
            [B, N, 4] predicted next states
        """
        # Project to hidden dim
        h = self.input_proj(x)

        # Process through layers
        for layer in self.layers:
            h = h + layer(h)

        # Particle attention (but no geometric operations)
        h = self.particle_attention(h)

        # Project to output
        delta = self.output_proj(h)

        return x + delta


class CliffordBlockNoBivector(nn.Module):
    """
    CliffordBlock WITHOUT geometric mixing (bivector generation).

    This ablation removes the outer product operations that generate
    bivector-like features, keeping only the standard MLP path.
    Used to test the importance of bivector components.
    """

    def __init__(self, dim: int, block_size: int, dropout: float = 0.1):
        super().__init__()

        self.dim = dim
        self.block_size = block_size

        # Main transformation (same as CliffordBlock)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

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
        # Standard MLP path only (NO geometric mixing)
        residual = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual

        # NOTE: Removed the geometric mixing path
        # No outer products, no bivector-like features
        x = self.norm2(x)

        return x


class CliffordNBodyNetNoBivector(nn.Module):
    """
    Ablation of CliffordNBodyNet WITHOUT bivector operations.

    Removes:
    - Geometric mixing in CliffordBlock (outer products)
    - GeometricInteractionLayer (position-velocity outer products)

    Keeps:
    - Mean-field aggregation
    - Residual connections
    - Same hidden dimension and layers

    This tests the hypothesis that bivector components are essential
    for capturing rotational dynamics in mean-field aggregation.
    """

    def __init__(
        self,
        n_particles: int = 5,
        hidden_dim: int = 128,
        block_size: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        use_mean_field: bool = True
    ):
        """
        Args:
            n_particles: Number of particles in the system
            hidden_dim: Hidden dimension
            block_size: Block size (kept for API compatibility)
            n_layers: Number of processing layers
            dropout: Dropout rate
            use_mean_field: If True (default), use mean-field aggregation
        """
        super().__init__()

        self.n_particles = n_particles
        self.block_size = block_size
        self.hidden_dim = hidden_dim
        self.use_mean_field = use_mean_field

        # Input projection
        self.input_proj = nn.Linear(4, hidden_dim)

        # Processing layers WITHOUT geometric mixing
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(CliffordBlockNoBivector(hidden_dim, block_size, dropout))

        # Mean-field aggregation (kept - this is what we're testing)
        if use_mean_field:
            self.mean_field = MeanFieldAggregationLayer(hidden_dim, n_components=8)

        # Simple feedforward (NO geometric interaction layer)
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

        # Process through blocks (NO geometric mixing)
        for layer in self.layers:
            h = layer(h)

        # Mean-field aggregation
        if self.use_mean_field:
            h = self.mean_field(h)

        # NOTE: No GeometricInteractionLayer here
        # (removed to test bivector importance)

        # Inter-particle interaction
        h = self.particle_interaction(h)

        # Project back to state space
        delta = self.output_proj(h)  # [B, N, 4]

        return x + delta


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the models
    print("Testing N-body Models...")

    B, N = 4, 5  # batch size, num particles
    x = torch.randn(B, N, 4)

    # Test Clifford model (default: no attention)
    clifford_model = CliffordNBodyNet(n_particles=N, hidden_dim=128)
    y_clifford = clifford_model(x)
    print(f"CliffordNBodyNet (no attention, default):")
    print(f"  Input: {x.shape}")
    print(f"  Output: {y_clifford.shape}")
    print(f"  Parameters: {count_parameters(clifford_model):,}")

    # Test Clifford model with attention
    clifford_attn = CliffordNBodyNet(n_particles=N, hidden_dim=128, use_attention=True)
    y_clifford_attn = clifford_attn(x)
    print(f"\nCliffordNBodyNet (with attention):")
    print(f"  Input: {x.shape}")
    print(f"  Output: {y_clifford_attn.shape}")
    print(f"  Parameters: {count_parameters(clifford_attn):,}")

    # Test baseline MLP
    baseline_model = BaselineNBodyNet(n_particles=N, hidden_dim=256)
    y_baseline = baseline_model(x)
    print(f"\nBaselineNBodyNet:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {y_baseline.shape}")
    print(f"  Parameters: {count_parameters(baseline_model):,}")

    # Test baseline with attention
    baseline_attn = BaselineNBodyNetWithAttention(n_particles=N, hidden_dim=128)
    y_baseline_attn = baseline_attn(x)
    print(f"\nBaselineNBodyNetWithAttention:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {y_baseline_attn.shape}")
    print(f"  Parameters: {count_parameters(baseline_attn):,}")

    # Test gradient flow
    loss = y_clifford.sum()
    loss.backward()
    print("\nGradient flow: OK")

    print("\nSUCCESS: All models working!")
