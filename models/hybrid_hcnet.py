"""
Core Hybrid HC-Net Architecture: Local MPNN + Global Clifford Mean-Field.

Key components:
- CliffordMeanField3DLayer: O(N) global aggregation in Cl(3,0) with trivector
- CliffordMeanField3DLayerExact: Uses actual Cl(3,0) geometric product
- CliffordBlock3DProposal: MLP + geometric mixing block
- HybridHCNet3D: Main hybrid architecture for N-body prediction
- HybridHCNet3DClassifier: Classification variant with global pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from nips_hcnet.layers.local_mpnn import LocalMPNNLayer


class CliffordMeanField3DLayer(nn.Module):
    """
    Global Cl(3,0) mean-field aggregation layer.

    Projects particle features to 8D multivector space (all grades of Cl(3,0):
    1 scalar + 3 vectors + 3 bivectors + 1 pseudoscalar).

    The trivector component (index 7) carries chirality information.

    Process:
    1. Project features to 8D multivector per particle: O(N)
    2. Compute mean multivector: mv_mean = mean(mv_per_particle): O(N)
    3. Outer product interaction: mv_i (x) mv_mean: O(N)
    4. Project interaction back and add residually: O(N)

    Total complexity: O(N)
    """

    def __init__(
        self,
        dim: int,
        n_components: int = 8,
        scale: float = 0.1,
    ):
        """
        Args:
            dim: Hidden dimension
            n_components: Multivector dimension (8 for Cl(3,0))
            scale: Weight of mean-field contribution
        """
        super().__init__()
        self.dim = dim
        self.n_components = n_components
        self.scale = scale

        # Project to multivector space
        self.particle_proj = nn.Linear(dim, n_components)
        self.meanfield_proj = nn.Linear(dim, n_components)

        # Project interaction back to hidden dim
        self.interaction_proj = nn.Linear(
            n_components * n_components, dim
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim] particle features

        Returns:
            [B, N, dim] updated features with global mean-field influence
        """
        B, N, D = x.shape

        # 1. Compute global mean-field: O(N)
        mean_field = x.mean(dim=1, keepdim=True)  # [B, 1, D]

        # 2. Project to multivector space
        particle_mv = self.particle_proj(x)  # [B, N, 8]
        meanfield_mv = self.meanfield_proj(mean_field)  # [B, 1, 8]
        meanfield_mv = meanfield_mv.expand(-1, N, -1)  # [B, N, 8]

        # 3. Outer product interaction (creates grade-mixed terms)
        # This captures bivector and trivector cross-terms
        interaction = torch.einsum(
            'bni,bnj->bnij', particle_mv, meanfield_mv
        )
        interaction = interaction.reshape(B, N, -1)  # [B, N, 64]

        # 4. Project back and add residually
        out = self.interaction_proj(interaction)  # [B, N, D]
        return self.norm(x + self.scale * out)


class CliffordMeanField3DLayerExact(nn.Module):
    """
    Exact Cl(3,0) mean-field using actual geometric product.

    Uses the precomputed Cayley tensor from CliffordAlgebra for
    principled multivector multiplication via einsum.

    More mathematically rigorous than CliffordMeanField3DLayer,
    but functionally similar.
    """

    def __init__(
        self,
        dim: int,
        scale: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.mv_dim = 8  # Cl(3,0)

        # Project to multivector space
        self.particle_proj = nn.Linear(dim, self.mv_dim)
        self.meanfield_proj = nn.Linear(dim, self.mv_dim)

        # Project product back
        self.output_proj = nn.Linear(self.mv_dim, dim)

        self.norm = nn.LayerNorm(dim)

        # We'll initialize the Cayley tensor lazily
        self._algebra = None

    def _get_algebra(self, device):
        if self._algebra is None or self._algebra.device != str(device):
            from nips_hcnet.algebra.clifford import CliffordAlgebra
            self._algebra = CliffordAlgebra(3, device=str(device))
        return self._algebra

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim]
        Returns:
            [B, N, dim]
        """
        B, N, D = x.shape
        algebra = self._get_algebra(x.device)

        # Project to multivector
        particle_mv = self.particle_proj(x)  # [B, N, 8]
        mean_field = x.mean(dim=1, keepdim=True)  # [B, 1, D]
        meanfield_mv = self.meanfield_proj(mean_field)  # [B, 1, 8]
        meanfield_mv = meanfield_mv.expand(-1, N, -1)  # [B, N, 8]

        # Geometric product via Cayley tensor
        product = algebra.geometric_product_optimized(
            particle_mv, meanfield_mv
        )  # [B, N, 8]

        # Project back
        out = self.output_proj(product)  # [B, N, D]
        return self.norm(x + self.scale * out)


class CliffordBlock3DProposal(nn.Module):
    """
    Processing block with MLP + geometric mixing.

    Follows the CliffordBlock3D pattern from hcnet/models/nbody_models_3d.py
    with group-wise outer products simulating higher-grade interactions.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()

        self.dim = dim
        self.group_size = 8
        self.n_groups = dim // self.group_size

        # MLP path
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

        # Geometric mixing
        self.geo_mix = nn.Linear(
            self.group_size * self.group_size, self.group_size
        )

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

        # MLP path
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

        # Outer product within groups -> bivector-like terms
        outer = torch.einsum('bngi,bngj->bngij', x_groups, x_groups)
        outer = outer.view(
            B, N, self.n_groups, self.group_size * self.group_size
        )

        geo_features = self.geo_mix(outer)
        geo_features = geo_features.view(B, N, D)

        x = self.norm2(residual + 0.1 * geo_features)
        return x


class HybridHCNet3D(nn.Module):
    """
    Main Hybrid HC-Net 3D Architecture.

    Combines local MPNN for precise nearby interactions with global
    Clifford mean-field for long-range collective effects.

    Per layer:
        local_out = LocalMPNNLayer(h, positions)
        global_out = CliffordMeanField3DLayer(h)
        fused = Fusion(cat(local_out, global_out))
        h = CliffordBlock3DProposal(fused)

    Total complexity: O(N) (both local and global are O(N))
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 4,
        k_neighbors: int = 10,
        cutoff: float = 5.0,
        n_rbf: int = 20,
        dropout: float = 0.1,
        use_exact_clifford: bool = False,
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            n_layers: Number of hybrid layers
            k_neighbors: kNN neighbors for local MPNN
            cutoff: Distance cutoff for local MPNN
            n_rbf: RBF centers for distance encoding
            dropout: Dropout rate
            use_exact_clifford: Use exact Cl(3,0) geometric product
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Input projection: [B, N, 6] -> [B, N, hidden_dim]
        self.input_proj = nn.Linear(6, hidden_dim)

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

            if use_exact_clifford:
                self.global_layers.append(
                    CliffordMeanField3DLayerExact(dim=hidden_dim)
                )
            else:
                self.global_layers.append(
                    CliffordMeanField3DLayer(dim=hidden_dim)
                )

            # Fusion: concatenate local + global -> project
            self.fusion_layers.append(nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

            self.clifford_blocks.append(
                CliffordBlock3DProposal(hidden_dim, dropout)
            )

        # Output projection: [B, N, hidden_dim] -> [B, N, 6]
        self.output_proj = nn.Linear(hidden_dim, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 6] particle states (x, y, z, vx, vy, vz)

        Returns:
            [B, N, 6] predicted next states (residual)
        """
        positions = x[:, :, :3]  # [B, N, 3]

        h = self.input_proj(x)  # [B, N, hidden_dim]

        for i in range(self.n_layers):
            # Local: message passing on kNN graph
            local_out = self.local_layers[i](h, positions)

            # Global: Clifford mean-field aggregation
            global_out = self.global_layers[i](h)

            # Fusion: combine local and global
            fused = torch.cat([local_out, global_out], dim=-1)
            fused = self.fusion_layers[i](fused)

            # Clifford processing block
            h = self.clifford_blocks[i](fused)

        # Output with residual
        delta = self.output_proj(h)  # [B, N, 6]
        return x + delta


class HybridHCNet3DClassifier(nn.Module):
    """
    Classification variant of HybridHCNet3D.

    Uses global pooling + MLP head for binary classification.
    Suitable for rotation direction and chirality classification tasks.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_classes: int = 2,
        k_neighbors: int = 10,
        cutoff: float = 5.0,
        dropout: float = 0.1,
        use_exact_clifford: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Input projection
        self.input_proj = nn.Linear(6, hidden_dim)

        # Hybrid layers
        self.local_layers = nn.ModuleList()
        self.global_layers = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        self.clifford_blocks = nn.ModuleList()

        for _ in range(n_layers):
            self.local_layers.append(
                LocalMPNNLayer(
                    hidden_dim=hidden_dim,
                    k_neighbors=k_neighbors,
                    cutoff=cutoff,
                    dropout=dropout,
                )
            )

            if use_exact_clifford:
                self.global_layers.append(
                    CliffordMeanField3DLayerExact(dim=hidden_dim)
                )
            else:
                self.global_layers.append(
                    CliffordMeanField3DLayer(dim=hidden_dim)
                )

            self.fusion_layers.append(nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

            self.clifford_blocks.append(
                CliffordBlock3DProposal(hidden_dim, dropout)
            )

        # Classification head: global pool + MLP
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 6] particle states

        Returns:
            [B, n_classes] class logits
        """
        positions = x[:, :, :3]

        h = self.input_proj(x)

        for i in range(self.n_layers):
            local_out = self.local_layers[i](h, positions)
            global_out = self.global_layers[i](h)

            fused = torch.cat([local_out, global_out], dim=-1)
            fused = self.fusion_layers[i](fused)
            h = self.clifford_blocks[i](fused)

        # Global mean pooling + classification
        h_pool = h.mean(dim=1)  # [B, hidden_dim]
        return self.classifier(h_pool)


if __name__ == '__main__':
    print("Testing Hybrid HC-Net 3D Models...")

    B, N = 4, 10
    x = torch.randn(B, N, 6)

    # Test HybridHCNet3D (regression)
    model = HybridHCNet3D(hidden_dim=128, n_layers=2, k_neighbors=5)
    y = model(x)
    loss = y.sum()
    loss.backward()
    model.zero_grad()
    print(f"  HybridHCNet3D: input={x.shape}, output={y.shape}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Test HybridHCNet3DClassifier
    clf = HybridHCNet3DClassifier(
        hidden_dim=128, n_layers=2, k_neighbors=5
    )
    logits = clf(x)
    loss = logits.sum()
    loss.backward()
    clf.zero_grad()
    print(f"  HybridHCNet3DClassifier: input={x.shape}, output={logits.shape}")

    # Test exact Clifford variant
    model_exact = HybridHCNet3D(
        hidden_dim=128, n_layers=2, k_neighbors=5, use_exact_clifford=True
    )
    y_exact = model_exact(x)
    print(f"  Exact Clifford variant: output={y_exact.shape}")

    # Test individual layers
    layer = CliffordMeanField3DLayer(dim=128)
    h = torch.randn(B, N, 128)
    h_out = layer(h)
    print(f"  CliffordMeanField3DLayer: {h.shape} -> {h_out.shape}")

    block = CliffordBlock3DProposal(dim=128)
    h_out2 = block(h)
    print(f"  CliffordBlock3DProposal: {h.shape} -> {h_out2.shape}")

    print("\nSUCCESS: All hybrid models working!")
