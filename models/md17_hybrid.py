"""
MD17-adapted Hybrid HC-Net for molecular force prediction.

Combines local MPNN (bond-scale interactions) with global Clifford
mean-field (long-range collective effects) for force prediction.

Input: positions [N, 3] + atomic numbers [N] -> forces [N, 3]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from nips_hcnet.layers.local_mpnn import LocalMPNNLayer
from nips_hcnet.models.hybrid_hcnet import (
    CliffordMeanField3DLayer,
    CliffordBlock3DProposal,
)


class MD17HybridHCNet(nn.Module):
    """
    Hybrid HC-Net adapted for MD17 molecular force prediction.

    Architecture:
    - Atom type embedding: nn.Embedding(n_atom_types, hidden//4)
    - Position embedding: Linear(3, hidden*3//4)
    - Per layer: LocalMPNN || CliffordMeanField3D -> Fusion -> CliffordBlock
    - Force output: Linear -> SiLU -> Linear(hidden, 3)

    Key adaptation from HybridHCNet3D:
    - No velocity input (only positions)
    - Atomic number embedding for chemical species
    - Force output instead of next-state prediction
    - Molecular-scale cutoff (5.0 Angstrom)
    """

    def __init__(
        self,
        n_atoms: int = 9,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_atom_types: int = 10,
        k_neighbors: int = 10,
        cutoff: float = 5.0,
        n_rbf: int = 20,
        dropout: float = 0.1,
    ):
        """
        Args:
            n_atoms: Number of atoms (fixed per molecule)
            hidden_dim: Hidden feature dimension
            n_layers: Number of hybrid layers
            n_atom_types: Max atomic number for embedding
            k_neighbors: kNN neighbors for local MPNN
            cutoff: Distance cutoff in Angstrom
            n_rbf: RBF centers for distance encoding
            dropout: Dropout rate
        """
        super().__init__()

        self.n_atoms = n_atoms
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Atom type embedding
        self.atom_embed = nn.Embedding(n_atom_types, hidden_dim // 4)

        # Position embedding
        self.pos_embed = nn.Linear(3, hidden_dim * 3 // 4)

        # Hybrid layers
        self.local_layers = nn.ModuleList()
        self.global_layers = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        self.clifford_blocks = nn.ModuleList()

        for _ in range(n_layers):
            self.local_layers.append(
                LocalMPNNLayer(
                    hidden_dim=hidden_dim,
                    n_rbf=n_rbf,
                    k_neighbors=k_neighbors,
                    cutoff=cutoff,
                    dropout=dropout,
                )
            )

            self.global_layers.append(
                CliffordMeanField3DLayer(dim=hidden_dim)
            )

            self.fusion_layers.append(nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ))

            self.clifford_blocks.append(
                CliffordBlock3DProposal(hidden_dim, dropout)
            )

        # Force output head
        self.force_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        positions: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            positions: [B, N, 3] atomic positions in Angstrom
            atomic_numbers: [B, N] atomic numbers (int)

        Returns:
            [B, N, 3] predicted forces
        """
        B, N, _ = positions.shape

        # Embed atoms: type + position
        atom_features = self.atom_embed(atomic_numbers)  # [B, N, hidden//4]
        pos_features = self.pos_embed(positions)  # [B, N, 3*hidden//4]
        h = torch.cat([atom_features, pos_features], dim=-1)  # [B, N, hidden]

        # Process through hybrid layers
        for i in range(self.n_layers):
            # Local: message passing on kNN molecular graph
            local_out = self.local_layers[i](h, positions)

            # Global: Clifford mean-field (captures long-range effects)
            global_out = self.global_layers[i](h)

            # Fusion
            fused = torch.cat([local_out, global_out], dim=-1)
            fused = self.fusion_layers[i](fused)

            # Clifford processing
            h = self.clifford_blocks[i](fused)

        # Predict forces
        forces = self.force_output(h)  # [B, N, 3]
        return forces


class MD17HybridHCNetEnergy(nn.Module):
    """
    Energy-conserving Hybrid HC-Net for MD17.

    Predicts per-atom energies, sums to total energy E, then computes
    forces as F_i = -dE/dr_i via autograd. This guarantees energy
    conservation by construction (as in NequIP, MACE, Allegro).

    Architecture is identical to MD17HybridHCNet except the output head
    predicts a scalar energy per atom instead of a 3D force vector.
    """

    def __init__(
        self,
        n_atoms: int = 9,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_atom_types: int = 10,
        k_neighbors: int = 10,
        cutoff: float = 5.0,
        n_rbf: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_atoms = n_atoms
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Atom type embedding
        self.atom_embed = nn.Embedding(n_atom_types, hidden_dim // 4)

        # Position embedding
        self.pos_embed = nn.Linear(3, hidden_dim * 3 // 4)

        # Hybrid layers
        self.local_layers = nn.ModuleList()
        self.global_layers = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        self.clifford_blocks = nn.ModuleList()

        for _ in range(n_layers):
            self.local_layers.append(
                LocalMPNNLayer(
                    hidden_dim=hidden_dim,
                    n_rbf=n_rbf,
                    k_neighbors=k_neighbors,
                    cutoff=cutoff,
                    dropout=dropout,
                )
            )

            self.global_layers.append(
                CliffordMeanField3DLayer(dim=hidden_dim)
            )

            self.fusion_layers.append(nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ))

            self.clifford_blocks.append(
                CliffordBlock3DProposal(hidden_dim, dropout)
            )

        # Energy output head: per-atom scalar energy
        self.energy_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        positions: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ):
        """
        Args:
            positions: [B, N, 3] atomic positions (must have requires_grad=True
                       for force computation)
            atomic_numbers: [B, N] atomic numbers (int)

        Returns:
            total_energy: [B] total energy per molecule
            forces: [B, N, 3] forces = -dE/dr
        """
        B, N, _ = positions.shape

        # Embed atoms: type + position
        atom_features = self.atom_embed(atomic_numbers)  # [B, N, hidden//4]
        pos_features = self.pos_embed(positions)  # [B, N, 3*hidden//4]
        h = torch.cat([atom_features, pos_features], dim=-1)  # [B, N, hidden]

        # Process through hybrid layers
        for i in range(self.n_layers):
            local_out = self.local_layers[i](h, positions)
            global_out = self.global_layers[i](h)
            fused = torch.cat([local_out, global_out], dim=-1)
            fused = self.fusion_layers[i](fused)
            h = self.clifford_blocks[i](fused)

        # Per-atom energies -> total energy
        atom_energies = self.energy_output(h).squeeze(-1)  # [B, N]
        total_energy = atom_energies.sum(dim=-1)  # [B]

        # Forces via autograd: F = -dE/dr
        grad_outputs = torch.ones_like(total_energy)
        forces_neg = torch.autograd.grad(
            total_energy,
            positions,
            grad_outputs=grad_outputs,
            create_graph=self.training,
            retain_graph=True,
        )[0]
        forces = -forces_neg  # [B, N, 3]

        return total_energy, forces


if __name__ == '__main__':
    print("Testing MD17 Hybrid HC-Net...")

    B, N = 4, 9  # Batch, atoms (ethanol)
    positions = torch.randn(B, N, 3)
    atomic_numbers = torch.randint(1, 9, (B, N))

    model = MD17HybridHCNet(
        n_atoms=N,
        hidden_dim=128,
        n_layers=4,
        k_neighbors=8,
        cutoff=5.0,
    )

    forces = model(positions, atomic_numbers)
    print(f"  Input: positions={positions.shape}, z={atomic_numbers.shape}")
    print(f"  Output forces: {forces.shape}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Test gradient flow
    loss = forces.sum()
    loss.backward()
    print(f"  Gradient flow: OK")

    # Test with different molecule sizes
    for n_atoms in [9, 12, 21]:  # ethanol, benzene, aspirin
        pos = torch.randn(2, n_atoms, 3)
        z = torch.randint(1, 9, (2, n_atoms))
        m = MD17HybridHCNet(n_atoms=n_atoms, hidden_dim=128, n_layers=2)
        f = m(pos, z)
        print(f"  N={n_atoms}: forces={f.shape}")

    # Test energy-conserving model
    print("\nTesting MD17 Hybrid HC-Net Energy...")
    pos_e = torch.randn(B, N, 3, requires_grad=True)
    z_e = torch.randint(1, 9, (B, N))
    model_e = MD17HybridHCNetEnergy(
        n_atoms=N, hidden_dim=128, n_layers=2,
        k_neighbors=8, cutoff=5.0,
    )
    energy, forces_e = model_e(pos_e, z_e)
    print(f"  Energy shape: {energy.shape}")
    print(f"  Forces shape: {forces_e.shape}")
    (energy.sum() + forces_e.sum()).backward()
    print(f"  Gradient flow: OK")

    print("\nSUCCESS: MD17 Hybrid HC-Net working!")
