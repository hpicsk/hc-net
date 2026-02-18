"""
MD17 Molecular Force Prediction Models.

Adapts N-body models for molecular dynamics force prediction.
Key differences from N-body:
- Variable atoms per molecule (fixed per molecule type)
- Atomic numbers as node features
- Force prediction instead of next-state prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MD17CliffordNet(nn.Module):
    """
    Clifford-inspired network for MD17 force prediction.

    Architecture similar to CliffordNBodyNet3D but adapted for:
    - Atomic number embeddings
    - Force prediction output
    - No velocity input (only positions)
    """

    def __init__(
        self,
        n_atoms: int = 9,  # Depends on molecule
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_atom_types: int = 10,  # Max atomic number (H=1, C=6, N=7, O=8, etc.)
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_atoms = n_atoms
        self.hidden_dim = hidden_dim

        # Atom type embedding
        self.atom_embed = nn.Embedding(n_atom_types, hidden_dim // 4)

        # Position embedding
        self.pos_embed = nn.Linear(3, hidden_dim * 3 // 4)

        # Processing layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(MD17CliffordBlock(hidden_dim, dropout))

        # Inter-atom attention
        self.atom_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True, dropout=dropout
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Geometric interaction
        self.geo_interaction = GeometricLayer(hidden_dim)

        # Force output
        self.force_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(
        self,
        positions: torch.Tensor,
        atomic_numbers: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            positions: [B, N, 3] atomic positions
            atomic_numbers: [B, N] atomic numbers

        Returns:
            [B, N, 3] predicted forces
        """
        B, N, _ = positions.shape

        # Embed atoms
        atom_features = self.atom_embed(atomic_numbers)  # [B, N, hidden/4]
        pos_features = self.pos_embed(positions)  # [B, N, 3*hidden/4]

        h = torch.cat([atom_features, pos_features], dim=-1)  # [B, N, hidden]

        # Process through Clifford blocks
        for layer in self.layers:
            h = layer(h)

        # Geometric interaction
        h = self.geo_interaction(h, positions)

        # Inter-atom attention
        attn_out, _ = self.atom_attention(h, h, h)
        h = self.attn_norm(h + attn_out)

        # Predict forces
        forces = self.force_output(h)

        return forces


class MD17CliffordBlock(nn.Module):
    """Processing block for MD17 model."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()

        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

        # Geometric mixing with fixed group size
        self.group_size = 8
        self.n_groups = dim // self.group_size
        self.geo_mix = nn.Linear(self.group_size * self.group_size, self.group_size)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        # MLP path
        residual = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual

        # Geometric mixing
        residual = x
        x_groups = x.view(B, N, self.n_groups, self.group_size)
        outer = torch.einsum('bngi,bngj->bngij', x_groups, x_groups)
        outer = outer.view(B, N, self.n_groups, self.group_size * self.group_size)
        geo_features = self.geo_mix(outer)
        geo_features = geo_features.view(B, N, D)

        x = self.norm2(residual + 0.1 * geo_features)

        return x


class GeometricLayer(nn.Module):
    """Computes geometric interactions using pairwise distances."""

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Edge network: processes pairwise features
        self.edge_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, N, hidden] node features
            pos: [B, N, 3] positions

        Returns:
            [B, N, hidden] updated features
        """
        B, N, _ = pos.shape

        # Compute pairwise distances
        pos_diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B, N, N, 3]
        dist = torch.norm(pos_diff, dim=-1, keepdim=True)  # [B, N, N, 1]

        # Build edge features
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)

        edge_input = torch.cat([h_i, h_j, dist], dim=-1)
        edge_features = self.edge_net(edge_input)

        # Aggregate messages
        msg = edge_features.mean(dim=2)  # [B, N, hidden]

        # Update
        update_input = torch.cat([h, msg], dim=-1)
        h_update = self.update_net(update_input)

        return self.norm(h + h_update)


class MD17BaselineNet(nn.Module):
    """
    Baseline MLP for MD17 without geometric operations.
    """

    def __init__(
        self,
        n_atoms: int = 9,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_atom_types: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_atoms = n_atoms
        self.hidden_dim = hidden_dim

        # Embeddings
        self.atom_embed = nn.Embedding(n_atom_types, hidden_dim // 4)
        self.pos_embed = nn.Linear(3, hidden_dim * 3 // 4)

        # Processing
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

        # Attention
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True, dropout=dropout
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Output
        self.output = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        positions: torch.Tensor,
        atomic_numbers: torch.Tensor
    ) -> torch.Tensor:
        B, N, _ = positions.shape

        # Embed
        atom_features = self.atom_embed(atomic_numbers)
        pos_features = self.pos_embed(positions)
        h = torch.cat([atom_features, pos_features], dim=-1)

        # Process
        for layer in self.layers:
            h = h + layer(h)

        # Attention
        attn_out, _ = self.attention(h, h, h)
        h = self.attn_norm(h + attn_out)

        # Output
        return self.output(h)


class MD17EGNNAdapter(nn.Module):
    """
    EGNN-style model adapted for MD17.

    Uses coordinate differences and distances for message passing.
    """

    def __init__(
        self,
        n_atoms: int = 9,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_atom_types: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_atoms = n_atoms

        # Embeddings
        self.atom_embed = nn.Embedding(n_atom_types, hidden_dim)

        # EGNN layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(EGNNForceLayer(hidden_dim, dropout))

        # Force output
        self.output = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        positions: torch.Tensor,
        atomic_numbers: torch.Tensor
    ) -> torch.Tensor:
        B, N, _ = positions.shape

        # Initialize node features from atom types
        h = self.atom_embed(atomic_numbers)

        # Process through EGNN layers
        for layer in self.layers:
            h = layer(h, positions)

        # Output forces
        return self.output(h)


class EGNNForceLayer(nn.Module):
    """EGNN layer for force prediction."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Edge MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        B, N, D = h.shape

        # Pairwise distances
        pos_diff = pos.unsqueeze(2) - pos.unsqueeze(1)
        dist_sq = (pos_diff ** 2).sum(dim=-1, keepdim=True)

        # Edge features
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)

        edge_input = torch.cat([h_i, h_j, dist_sq], dim=-1)
        m_ij = self.edge_mlp(edge_input)

        # Aggregate
        m_i = m_ij.mean(dim=2)

        # Update
        h_out = self.node_mlp(torch.cat([h, m_i], dim=-1))
        h_out = self.dropout(h_out)

        return self.norm(h + h_out)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("Testing MD17 Models...")

    B, N = 4, 9  # Batch size, atoms (ethanol)
    positions = torch.randn(B, N, 3)
    atomic_numbers = torch.randint(1, 9, (B, N))

    # Test Clifford model
    print("\n1. MD17CliffordNet:")
    clifford_model = MD17CliffordNet(n_atoms=N, hidden_dim=128)
    forces = clifford_model(positions, atomic_numbers)
    print(f"   Input positions: {positions.shape}")
    print(f"   Output forces: {forces.shape}")
    print(f"   Parameters: {count_parameters(clifford_model):,}")

    # Test baseline
    print("\n2. MD17BaselineNet:")
    baseline_model = MD17BaselineNet(n_atoms=N, hidden_dim=128)
    forces_baseline = baseline_model(positions, atomic_numbers)
    print(f"   Output forces: {forces_baseline.shape}")
    print(f"   Parameters: {count_parameters(baseline_model):,}")

    # Test EGNN adapter
    print("\n3. MD17EGNNAdapter:")
    egnn_model = MD17EGNNAdapter(n_atoms=N, hidden_dim=128)
    forces_egnn = egnn_model(positions, atomic_numbers)
    print(f"   Output forces: {forces_egnn.shape}")
    print(f"   Parameters: {count_parameters(egnn_model):,}")

    # Test gradient flow
    loss = forces.sum()
    loss.backward()
    print("\nGradient flow: OK")

    print("\nSUCCESS: All MD17 models working!")
