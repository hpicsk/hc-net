"""
Ablation Study Framework.

Systematically evaluates contribution of each component in HC-Net:
1. Geometric mixing (bivector-like operations)
2. Inter-particle attention
3. Residual connections
4. Layer normalization
5. Number of layers
6. Hidden dimension

Addresses reviewer concern: "No ablation studies to isolate component contributions"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
import argparse
import sys
from importlib.util import spec_from_file_location, module_from_spec

_PCNN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PCNN_ROOT not in sys.path:
    sys.path.insert(0, _PCNN_ROOT)


def _import_module(name, path):
    """Import module directly."""
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import data
_nbody_dataset = _import_module(
    "nbody_dataset", os.path.join(_PCNN_ROOT, 'data', 'nbody_dataset.py')
)
get_nbody_loaders_with_ood = _nbody_dataset.get_nbody_loaders_with_ood


@dataclass
class AblationConfig:
    """Configuration for ablation study."""
    # Base settings
    n_particles: int = 5
    hidden_dim: int = 128
    n_layers: int = 4
    dropout: float = 0.1

    # Training
    epochs: int = 100
    batch_size: int = 128
    lr: float = 0.001
    weight_decay: float = 1e-5
    n_train: int = 1000
    n_test: int = 500

    # Experiment
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    device: str = 'cuda'
    save_dir: str = './results/ablation'


# ============================================================================
# Ablation Model Variants
# ============================================================================

class HCNetFull(nn.Module):
    """Full HC-Net model (baseline for ablation, default: no attention)."""

    def __init__(self, n_particles: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.n_particles = n_particles
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(4, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(HCNetBlock(hidden_dim, dropout, use_geo_mixing=True))

        self.geo_interaction = GeometricInteraction(hidden_dim)
        # Default: feedforward (matches CliffordNBodyNet use_attention=False)
        self.particle_interaction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.output_proj = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        h = self.geo_interaction(h)
        h = self.particle_interaction(h)
        delta = self.output_proj(h)
        return x + delta


class HCNetNoGeoMixing(nn.Module):
    """HC-Net without geometric mixing in blocks."""

    def __init__(self, n_particles: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.n_particles = n_particles
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(4, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(HCNetBlock(hidden_dim, dropout, use_geo_mixing=False))

        self.geo_interaction = GeometricInteraction(hidden_dim)
        self.particle_interaction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.output_proj = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        h = self.geo_interaction(h)
        h = self.particle_interaction(h)
        delta = self.output_proj(h)
        return x + delta


class HCNetNoGeoInteraction(nn.Module):
    """HC-Net without global geometric interaction layer."""

    def __init__(self, n_particles: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.n_particles = n_particles
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(4, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(HCNetBlock(hidden_dim, dropout, use_geo_mixing=True))

        # No geo_interaction
        self.particle_interaction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.output_proj = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        # Skip geo_interaction
        h = self.particle_interaction(h)
        delta = self.output_proj(h)
        return x + delta


class HCNetWithAttention(nn.Module):
    """HC-Net WITH particle attention (attention is OFF by default).

    This variant adds O(N^2) multi-head attention back to test whether
    it helps or hurts. Previous results showed it degrades performance.
    """

    def __init__(self, n_particles: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.n_particles = n_particles
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(4, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(HCNetBlock(hidden_dim, dropout, use_geo_mixing=True))

        self.geo_interaction = GeometricInteraction(hidden_dim)
        # Add attention back (O(N^2), expected to hurt performance)
        self.particle_attention = ParticleAttention(hidden_dim, dropout)
        self.output_proj = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        h = self.geo_interaction(h)
        h = self.particle_attention(h)
        delta = self.output_proj(h)
        return x + delta


class HCNetNoResidual(nn.Module):
    """HC-Net without residual connections in blocks."""

    def __init__(self, n_particles: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.n_particles = n_particles
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(4, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(HCNetBlockNoResidual(hidden_dim, dropout))

        self.geo_interaction = GeometricInteraction(hidden_dim)
        self.particle_interaction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.output_proj = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        h = self.geo_interaction(h)
        h = self.particle_interaction(h)
        delta = self.output_proj(h)
        return x + delta


class HCNetNoLayerNorm(nn.Module):
    """HC-Net without layer normalization."""

    def __init__(self, n_particles: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.n_particles = n_particles
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(4, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(HCNetBlockNoNorm(hidden_dim, dropout))

        self.geo_interaction = GeometricInteractionNoNorm(hidden_dim)
        self.particle_interaction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.output_proj = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        h = self.geo_interaction(h)
        h = self.particle_interaction(h)
        delta = self.output_proj(h)
        return x + delta


class HCNetWithMeanField(nn.Module):
    """
    HC-Net with mean-field aggregation for O(N) inter-particle communication.

    This variant adds global mean-field interaction, allowing particles to
    communicate through the collective system state without O(N^2) pairwise
    message passing. Enables true N-body interaction modeling at O(N) cost.
    """

    def __init__(self, n_particles: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.n_particles = n_particles
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(4, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(HCNetBlock(hidden_dim, dropout, use_geo_mixing=True))

        # Mean-field aggregation for O(N) inter-particle communication
        self.mean_field = MeanFieldAggregation(hidden_dim, n_components=8)

        self.geo_interaction = GeometricInteraction(hidden_dim)
        self.particle_interaction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.output_proj = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        # Mean-field aggregation: O(N) inter-particle communication
        h = self.mean_field(h)
        h = self.geo_interaction(h)
        h = self.particle_interaction(h)
        delta = self.output_proj(h)
        return x + delta


# ============================================================================
# Building Blocks
# ============================================================================

class HCNetBlock(nn.Module):
    """Standard HC-Net processing block."""

    def __init__(self, dim: int, dropout: float, use_geo_mixing: bool = True):
        super().__init__()
        self.use_geo_mixing = use_geo_mixing

        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

        if use_geo_mixing:
            self.group_size = 8
            self.n_groups = dim // self.group_size
            self.geo_mix = nn.Linear(self.group_size * self.group_size, self.group_size)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        # MLP path with residual
        residual = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual

        # Geometric mixing path
        if self.use_geo_mixing:
            residual = x
            x_groups = x.view(B, N, self.n_groups, self.group_size)
            outer = torch.einsum('bngi,bngj->bngij', x_groups, x_groups)
            outer = outer.view(B, N, self.n_groups, self.group_size * self.group_size)
            geo_features = self.geo_mix(outer)
            geo_features = geo_features.view(B, N, D)
            x = self.norm2(residual + 0.1 * geo_features)

        return x


class HCNetBlockNoResidual(nn.Module):
    """HC-Net block without residual connections."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()

        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

        self.group_size = 8
        self.n_groups = dim // self.group_size
        self.geo_mix = nn.Linear(self.group_size * self.group_size, self.group_size)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        # MLP path (no residual)
        x = self.norm1(x)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        # Geometric mixing (no residual)
        x_groups = x.view(B, N, self.n_groups, self.group_size)
        outer = torch.einsum('bngi,bngj->bngij', x_groups, x_groups)
        outer = outer.view(B, N, self.n_groups, self.group_size * self.group_size)
        geo_features = self.geo_mix(outer)
        x = self.norm2(geo_features.view(B, N, D))

        return x


class HCNetBlockNoNorm(nn.Module):
    """HC-Net block without layer normalization."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()

        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

        self.group_size = 8
        self.n_groups = dim // self.group_size
        self.geo_mix = nn.Linear(self.group_size * self.group_size, self.group_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        residual = x
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x_groups = x.view(B, N, self.n_groups, self.group_size)
        outer = torch.einsum('bngi,bngj->bngij', x_groups, x_groups)
        outer = outer.view(B, N, self.n_groups, self.group_size * self.group_size)
        geo_features = self.geo_mix(outer)
        x = residual + 0.1 * geo_features.view(B, N, D)

        return x


class GeometricInteraction(nn.Module):
    """Geometric interaction layer."""

    def __init__(self, dim: int):
        super().__init__()
        self.n_components = 8
        self.pos_proj = nn.Linear(dim, self.n_components)
        self.vel_proj = nn.Linear(dim, self.n_components)
        self.interaction_proj = nn.Linear(self.n_components * self.n_components, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = self.pos_proj(x)
        vel = self.vel_proj(x)
        interaction = torch.einsum('bni,bnj->bnij', pos, vel)
        interaction = interaction.view(*x.shape[:-1], -1)
        interaction = self.interaction_proj(interaction)
        return self.norm(x + interaction)


class GeometricInteractionNoNorm(nn.Module):
    """Geometric interaction without normalization."""

    def __init__(self, dim: int):
        super().__init__()
        self.n_components = 8
        self.pos_proj = nn.Linear(dim, self.n_components)
        self.vel_proj = nn.Linear(dim, self.n_components)
        self.interaction_proj = nn.Linear(self.n_components * self.n_components, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = self.pos_proj(x)
        vel = self.vel_proj(x)
        interaction = torch.einsum('bni,bnj->bnij', pos, vel)
        interaction = interaction.view(*x.shape[:-1], -1)
        interaction = self.interaction_proj(interaction)
        return x + interaction


class MeanFieldAggregation(nn.Module):
    """
    Mean-field aggregation for O(N) inter-particle communication.

    Physics: Each particle interacts with the collective mean-field state,
    approximating gravitational mean-field theory.
    """

    def __init__(self, dim: int, n_components: int = 8, scale: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_components = n_components
        self.scale = scale

        self.particle_proj = nn.Linear(dim, n_components)
        self.meanfield_proj = nn.Linear(dim, n_components)
        self.interaction_proj = nn.Linear(n_components * n_components, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        # Global mean-field: O(N)
        mean_field = x.mean(dim=1, keepdim=True)  # [B, 1, dim]

        # Project to interaction space
        particle_feat = self.particle_proj(x)            # [B, N, n_comp]
        meanfield_feat = self.meanfield_proj(mean_field)  # [B, 1, n_comp]

        # Outer product interaction: O(N)
        # Broadcast mean-field to all particles then compute outer product
        meanfield_feat = meanfield_feat.expand(-1, N, -1)  # [B, N, n_comp]
        interaction = torch.einsum('bni,bnj->bnij', particle_feat, meanfield_feat)
        interaction = interaction.view(B, N, -1)  # [B, N, n_comp²]

        # Project back and residual
        out = self.interaction_proj(interaction)
        return self.norm(x + self.scale * out)


class ParticleAttention(nn.Module):
    """Particle attention layer."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=4, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class ParticleAttentionNoNorm(nn.Module):
    """Particle attention without normalization."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=4, batch_first=True, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = x + self.ff(x)
        return x


# ============================================================================
# Training and Evaluation
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_input, batch_target in train_loader:
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)

        optimizer.zero_grad()
        output = model(batch_input)
        loss = nn.functional.mse_loss(output, batch_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, test_loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                batch_input, batch_target = batch
            else:
                batch_input, batch_target, _ = batch

            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            output = model(batch_input)
            loss = nn.functional.mse_loss(output, batch_target)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def run_ablation_experiment(
    model_class,
    model_name: str,
    seed: int,
    config: AblationConfig
) -> Dict:
    """Run single ablation experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = config.device if torch.cuda.is_available() else 'cpu'

    # Data
    train_loader, test_loader, ood_loaders = get_nbody_loaders_with_ood(
        n_train=config.n_train,
        n_test=config.n_test,
        n_particles=config.n_particles,
        rotation_angles=[0, 45, 90],
        batch_size=config.batch_size,
        seed=seed
    )

    # Model
    model = model_class(
        n_particles=config.n_particles,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        dropout=config.dropout
    ).to(device)

    n_params = count_parameters(model)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Train
    best_test_loss = float('inf')
    start_time = time.time()

    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_loss = evaluate(model, test_loader, device)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = model.state_dict().copy()

        scheduler.step()

    training_time = time.time() - start_time

    # Load best and evaluate OOD
    model.load_state_dict(best_state)
    ood_losses = {angle: evaluate(model, loader, device) for angle, loader in ood_loaders.items()}

    return {
        'model': model_name,
        'seed': seed,
        'n_params': n_params,
        'best_test_loss': best_test_loss,
        'ood_losses': ood_losses,
        'training_time': training_time
    }


def run_full_ablation(config: AblationConfig) -> Dict:
    """Run complete ablation study."""
    os.makedirs(config.save_dir, exist_ok=True)

    # Define ablation variants
    # Note: 'full' = default HC-Net (no attention, O(N) complexity)
    # 'with_attention' adds O(N^2) attention back to test its effect
    ablation_variants = {
        'full': HCNetFull,
        'no_geo_mixing': HCNetNoGeoMixing,
        'no_geo_interaction': HCNetNoGeoInteraction,
        'with_attention': HCNetWithAttention,
        'no_residual': HCNetNoResidual,
        'no_layer_norm': HCNetNoLayerNorm,
        'with_mean_field': HCNetWithMeanField,
    }

    all_results = []

    print("Running Ablation Study")
    print(f"Variants: {list(ablation_variants.keys())}")
    print(f"Seeds: {config.seeds}")
    print("=" * 60)

    for model_name, model_class in ablation_variants.items():
        for seed in config.seeds:
            print(f"\n{model_name}, seed={seed}...")

            try:
                result = run_ablation_experiment(model_class, model_name, seed, config)
                all_results.append(result)
                print(f"  Test MSE: {result['best_test_loss']:.6f}, OOD 90°: {result['ood_losses'].get(90, 'N/A'):.6f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results.append({'model': model_name, 'seed': seed, 'error': str(e)})

    # Layer count ablation
    print("\n--- Layer Count Ablation ---")
    for n_layers in [1, 2, 4, 6, 8]:
        config_copy = AblationConfig(**asdict(config))
        config_copy.n_layers = n_layers

        for seed in config.seeds[:1]:  # Just one seed for layer ablation
            print(f"\nn_layers={n_layers}, seed={seed}...")
            try:
                result = run_ablation_experiment(HCNetFull, f'layers_{n_layers}', seed, config_copy)
                result['n_layers'] = n_layers
                all_results.append(result)
                print(f"  Test MSE: {result['best_test_loss']:.6f}")
            except Exception as e:
                print(f"  ERROR: {e}")

    # Hidden dim ablation
    print("\n--- Hidden Dimension Ablation ---")
    for hidden_dim in [32, 64, 128, 256]:
        config_copy = AblationConfig(**asdict(config))
        config_copy.hidden_dim = hidden_dim

        for seed in config.seeds[:1]:
            print(f"\nhidden_dim={hidden_dim}, seed={seed}...")
            try:
                result = run_ablation_experiment(HCNetFull, f'hidden_{hidden_dim}', seed, config_copy)
                result['hidden_dim'] = hidden_dim
                all_results.append(result)
                print(f"  Test MSE: {result['best_test_loss']:.6f}")
            except Exception as e:
                print(f"  ERROR: {e}")

    # Aggregate and save
    summary = aggregate_ablation_results(all_results, ablation_variants.keys())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(config.save_dir, f"ablation_{timestamp}.json")

    with open(results_file, 'w') as f:
        json.dump({'config': asdict(config), 'results': all_results, 'summary': summary}, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")
    print_ablation_summary(summary)

    return {'results': all_results, 'summary': summary}


def aggregate_ablation_results(results: List[Dict], variant_names) -> Dict:
    """Aggregate ablation results."""
    summary = {}

    for variant in variant_names:
        variant_results = [r for r in results if r.get('model') == variant and 'error' not in r]
        if variant_results:
            test_losses = [r['best_test_loss'] for r in variant_results]
            summary[variant] = {
                'test_mse_mean': np.mean(test_losses),
                'test_mse_std': np.std(test_losses),
                'n_params': variant_results[0]['n_params']
            }

    return summary


def print_ablation_summary(summary: Dict):
    """Print ablation summary."""
    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)

    if 'full' in summary:
        full_mse = summary['full']['test_mse_mean']
        print(f"\nBase model (full, no attention): {full_mse:.6f} MSE")
        print("\nComponent Ablation:")
        print(f"  - Removing a component: higher MSE = component helps")
        print(f"  - Adding attention: higher MSE = attention hurts")
        print(f"\n{'Variant':<20} {'Test MSE':>12} {'Δ MSE':>12} {'% Change':>10}")
        print("-" * 55)

        for variant, data in summary.items():
            delta = data['test_mse_mean'] - full_mse
            pct_change = (delta / full_mse) * 100
            print(f"{variant:<20} {data['test_mse_mean']:>12.6f} {delta:>+12.6f} {pct_change:>+9.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Ablation Study')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n-train', type=int, default=1000)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-dir', type=str, default='./results/ablation')

    args = parser.parse_args()

    config = AblationConfig(
        epochs=args.epochs,
        n_train=args.n_train,
        seeds=args.seeds,
        device=args.device,
        save_dir=args.save_dir
    )

    run_full_ablation(config)


if __name__ == '__main__':
    main()
