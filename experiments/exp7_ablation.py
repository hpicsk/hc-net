"""
Experiment 7: Ablation Study on MD17 Ethanol.

Systematically evaluates contribution of each HC-Net component on a real
molecular dynamics task where architectural choices matter.

Ablation variants:
1. full:           Complete HC-Net (local MPNN + global mean-field + geo mixing)
2. no_local:       Remove local MPNN (global-only, no neighbor message passing)
3. no_global:      Remove global mean-field (local-only, no collective effects)
4. no_geo_mixing:  Remove geometric product in CliffordBlock
5. no_residual:    Remove skip connections in CliffordBlock
6. no_layer_norm:  Remove normalization in CliffordBlock

Also runs layer count ablation (1,2,4,6,8) and hidden dim ablation (32,64,128,256).

Usage:
    python -m nips_hcnet.train --experiment exp7 --seeds 42 --epochs 100
"""

import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from nips_hcnet.layers.local_mpnn import LocalMPNNLayer
from nips_hcnet.models.hybrid_hcnet import (
    CliffordMeanField3DLayer,
)
from nips_hcnet.data.md17_dataset import MD17_MOLECULES, get_md17_loaders


# ============================================================================
# Ablation-Aware Clifford Block
# ============================================================================

class AblationCliffordBlock(nn.Module):
    """CliffordBlock3DProposal with toggleable geo mixing, residual, norm."""

    def __init__(self, dim, dropout=0.1,
                 use_geo_mixing=True, use_residual=True, use_layer_norm=True):
        super().__init__()
        self.dim = dim
        self.use_geo_mixing = use_geo_mixing
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm

        self.group_size = 8
        self.n_groups = dim // self.group_size

        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

        if use_geo_mixing:
            self.geo_mix = nn.Linear(
                self.group_size * self.group_size, self.group_size
            )

        if use_layer_norm:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, D = x.shape

        # MLP path
        residual = x
        if self.use_layer_norm:
            x = self.norm1(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + residual

        # Geometric mixing path
        if self.use_geo_mixing:
            residual = x
            x_groups = x.view(B, N, self.n_groups, self.group_size)
            outer = torch.einsum('bngi,bngj->bngij', x_groups, x_groups)
            outer = outer.view(B, N, self.n_groups,
                               self.group_size * self.group_size)
            geo_features = self.geo_mix(outer).view(B, N, D)
            if self.use_residual:
                x = geo_features * 0.1 + residual
            else:
                x = geo_features
            if self.use_layer_norm:
                x = self.norm2(x)

        return x


# ============================================================================
# Ablation HC-Net for MD17
# ============================================================================

class MD17AblationHCNet(nn.Module):
    """
    MD17 force prediction model with toggleable components for ablation.

    Architecture mirrors MD17HybridHCNet but each component can be disabled:
    - use_local: LocalMPNN layers for neighbor message passing
    - use_global: CliffordMeanField3DLayer for global mean-field
    - use_geo_mixing: Geometric product in CliffordBlock
    - use_residual: Skip connections
    - use_layer_norm: Layer normalization
    """

    def __init__(
        self,
        n_atoms=9,
        hidden_dim=128,
        n_layers=4,
        k_neighbors=8,
        cutoff=5.0,
        n_rbf=20,
        dropout=0.1,
        use_local=True,
        use_global=True,
        use_geo_mixing=True,
        use_residual=True,
        use_layer_norm=True,
    ):
        super().__init__()
        self.n_atoms = n_atoms
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.use_local = use_local
        self.use_global = use_global

        # Atom type + position embedding
        self.atom_embed = nn.Embedding(10, hidden_dim // 4)
        self.pos_embed = nn.Linear(3, hidden_dim * 3 // 4)

        # Per-layer components
        if use_local:
            self.local_layers = nn.ModuleList([
                LocalMPNNLayer(
                    hidden_dim=hidden_dim, n_rbf=n_rbf,
                    k_neighbors=k_neighbors, cutoff=cutoff, dropout=dropout
                ) for _ in range(n_layers)
            ])

        if use_global:
            self.global_layers = nn.ModuleList([
                CliffordMeanField3DLayer(dim=hidden_dim) for _ in range(n_layers)
            ])

        # Fusion: depends on which branches are active
        if use_local and use_global:
            fusion_in = hidden_dim * 2
        else:
            fusion_in = hidden_dim
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_in, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ) for _ in range(n_layers)
        ])

        self.clifford_blocks = nn.ModuleList([
            AblationCliffordBlock(
                hidden_dim, dropout,
                use_geo_mixing=use_geo_mixing,
                use_residual=use_residual,
                use_layer_norm=use_layer_norm,
            ) for _ in range(n_layers)
        ])

        self.force_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, positions, atomic_numbers):
        B, N, _ = positions.shape

        atom_features = self.atom_embed(atomic_numbers)
        pos_features = self.pos_embed(positions)
        h = torch.cat([atom_features, pos_features], dim=-1)

        for i in range(self.n_layers):
            parts = []
            if self.use_local:
                parts.append(self.local_layers[i](h, positions))
            if self.use_global:
                parts.append(self.global_layers[i](h))

            if len(parts) == 2:
                fused = torch.cat(parts, dim=-1)
            elif len(parts) == 1:
                fused = parts[0]
            else:
                fused = h  # no local or global — just pass through

            fused = self.fusion_layers[i](fused)
            h = self.clifford_blocks[i](fused)

        return self.force_output(h)


# ============================================================================
# Training and Evaluation
# ============================================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    n_batches = 0

    for batch_pos, batch_z, batch_forces in loader:
        batch_pos = batch_pos.to(device)
        batch_z = batch_z.to(device)
        batch_forces = batch_forces.to(device)

        optimizer.zero_grad()
        pred = model(batch_pos, batch_z)
        loss = criterion(pred, batch_forces)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mae += (pred - batch_forces).abs().mean().item()
        n_batches += 1

    return total_loss / n_batches, total_mae / n_batches


def evaluate(model, loader, device):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_pos, batch_z, batch_forces in loader:
            batch_pos = batch_pos.to(device)
            batch_z = batch_z.to(device)
            batch_forces = batch_forces.to(device)

            pred = model(batch_pos, batch_z)
            total_mse += ((pred - batch_forces) ** 2).mean().item()
            total_mae += (pred - batch_forces).abs().mean().item()
            n_batches += 1

    return total_mse / n_batches, total_mae / n_batches


# ============================================================================
# Ablation Experiment
# ============================================================================

@dataclass
class AblationConfig:
    molecule: str = 'ethanol'
    n_train: int = 1000
    n_val: int = 200
    n_test: int = 500
    hidden_dim: int = 128
    n_layers: int = 4
    epochs: int = 100
    batch_size: int = 32
    lr: float = 0.001
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    device: str = 'cuda'
    save_dir: str = './results/ablation'


# Define ablation variants
ABLATION_VARIANTS = {
    'full':           dict(use_local=True,  use_global=True,  use_geo_mixing=True,  use_residual=True,  use_layer_norm=True),
    'no_local':       dict(use_local=False, use_global=True,  use_geo_mixing=True,  use_residual=True,  use_layer_norm=True),
    'no_global':      dict(use_local=True,  use_global=False, use_geo_mixing=True,  use_residual=True,  use_layer_norm=True),
    'no_geo_mixing':  dict(use_local=True,  use_global=True,  use_geo_mixing=False, use_residual=True,  use_layer_norm=True),
    'no_residual':    dict(use_local=True,  use_global=True,  use_geo_mixing=True,  use_residual=False, use_layer_norm=True),
    'no_layer_norm':  dict(use_local=True,  use_global=True,  use_geo_mixing=True,  use_residual=True,  use_layer_norm=False),
}


def run_single_ablation(
    variant_name, variant_flags, config, seed,
    n_layers_override=None, hidden_dim_override=None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    n_atoms = MD17_MOLECULES[config.molecule]['n_atoms']
    hidden_dim = hidden_dim_override or config.hidden_dim
    n_layers = n_layers_override or config.n_layers

    train_loader, val_loader, test_loader = get_md17_loaders(
        molecule=config.molecule,
        n_train=config.n_train, n_val=config.n_val, n_test=config.n_test,
        batch_size=config.batch_size, num_workers=0, seed=seed,
    )

    model = MD17AblationHCNet(
        n_atoms=n_atoms,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        k_neighbors=min(8, n_atoms - 1),
        cutoff=5.0,
        **variant_flags,
    ).to(device)

    n_params = count_parameters(model)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    best_val_mse = float('inf')
    best_state = None
    epoch_times = []

    for epoch in range(config.epochs):
        t0 = time.time()
        train_loss, train_mae = train_epoch(model, train_loader, optimizer, criterion, device)
        val_mse, val_mae = evaluate(model, val_loader, device)
        epoch_times.append(time.time() - t0)
        scheduler.step(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 25 == 0:
            print(f"      Epoch {epoch+1}: val_mse={val_mse:.6f}, val_mae={val_mae:.6f}")

    # Load best and test
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_mse, test_mae = evaluate(model, test_loader, device)

    return {
        'variant': variant_name,
        'seed': seed,
        'n_params': n_params,
        'test_force_mse': test_mse,
        'test_force_mae': test_mae,
        'best_val_mse': best_val_mse,
        'mean_epoch_time': float(np.mean(epoch_times)),
        'hidden_dim': hidden_dim,
        'n_layers': n_layers,
    }


def run_full_ablation(config):
    os.makedirs(config.save_dir, exist_ok=True)
    all_results = []

    print("=" * 70)
    print("EXPERIMENT 7: ABLATION STUDY ON MD17 ETHANOL")
    print("=" * 70)
    print(f"Molecule: {config.molecule}")
    print(f"Train/Val/Test: {config.n_train}/{config.n_val}/{config.n_test}")
    print(f"Epochs: {config.epochs}, Seeds: {config.seeds}")
    print(f"Variants: {list(ABLATION_VARIANTS.keys())}")
    print()

    # --- Component ablation ---
    print("--- Component Ablation ---")
    for variant_name, flags in ABLATION_VARIANTS.items():
        for seed in config.seeds:
            print(f"\n  {variant_name}, seed={seed}...")
            try:
                result = run_single_ablation(variant_name, flags, config, seed)
                all_results.append(result)
                print(f"    test_mse={result['test_force_mse']:.6f}, "
                      f"test_mae={result['test_force_mae']:.6f}, "
                      f"params={result['n_params']:,}")
            except Exception as e:
                print(f"    ERROR: {e}")
                all_results.append({
                    'variant': variant_name, 'seed': seed, 'error': str(e)
                })

    # --- Layer count ablation ---
    print("\n--- Layer Count Ablation ---")
    full_flags = ABLATION_VARIANTS['full']
    for n_layers in [1, 2, 4, 6, 8]:
        for seed in config.seeds[:1]:  # Single seed for hyperparameter sweeps
            variant_name = f'layers_{n_layers}'
            print(f"\n  {variant_name}, seed={seed}...")
            try:
                result = run_single_ablation(
                    variant_name, full_flags, config, seed,
                    n_layers_override=n_layers,
                )
                all_results.append(result)
                print(f"    test_mse={result['test_force_mse']:.6f}")
            except Exception as e:
                print(f"    ERROR: {e}")

    # --- Hidden dim ablation ---
    print("\n--- Hidden Dimension Ablation ---")
    for hidden_dim in [32, 64, 128, 256]:
        for seed in config.seeds[:1]:
            variant_name = f'hidden_{hidden_dim}'
            print(f"\n  {variant_name}, seed={seed}...")
            try:
                result = run_single_ablation(
                    variant_name, full_flags, config, seed,
                    hidden_dim_override=hidden_dim,
                )
                all_results.append(result)
                print(f"    test_mse={result['test_force_mse']:.6f}")
            except Exception as e:
                print(f"    ERROR: {e}")

    # --- Aggregate and report ---
    summary = aggregate_results(all_results)
    print_summary(summary)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(config.save_dir, f"ablation_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump({
            'config': asdict(config),
            'results': all_results,
            'summary': summary,
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    return {'results': all_results, 'summary': summary}


def aggregate_results(results):
    summary = {}
    # Group by variant name
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        if 'error' not in r:
            groups[r['variant']].append(r)

    for variant, runs in groups.items():
        mse_vals = [r['test_force_mse'] for r in runs]
        mae_vals = [r['test_force_mae'] for r in runs]
        summary[variant] = {
            'test_mse_mean': float(np.mean(mse_vals)),
            'test_mse_std': float(np.std(mse_vals)),
            'test_mae_mean': float(np.mean(mae_vals)),
            'test_mae_std': float(np.std(mae_vals)),
            'n_params': runs[0]['n_params'],
            'n_seeds': len(runs),
        }
    return summary


def print_summary(summary):
    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY (MD17 Ethanol)")
    print("=" * 70)

    # Component ablation
    if 'full' in summary:
        full_mse = summary['full']['test_mse_mean']
        full_mae = summary['full']['test_mae_mean']

        print(f"\nBase model (full): MSE={full_mse:.6f}, MAE={full_mae:.6f}")
        print(f"\n{'Variant':<18} {'Test MSE':>14} {'Test MAE':>14} "
              f"{'Delta MSE':>12} {'% Change':>10} {'Params':>10}")
        print("-" * 80)

        for variant in ABLATION_VARIANTS:
            if variant in summary:
                d = summary[variant]
                delta = d['test_mse_mean'] - full_mse
                pct = (delta / full_mse) * 100 if full_mse > 0 else 0
                print(f"{variant:<18} "
                      f"{d['test_mse_mean']:>8.6f}+/-{d['test_mse_std']:<4.4f} "
                      f"{d['test_mae_mean']:>8.6f}+/-{d['test_mae_std']:<4.4f} "
                      f"{delta:>+12.6f} {pct:>+9.1f}% "
                      f"{d['n_params']:>10,}")

    # Layer count
    layer_variants = sorted(
        [k for k in summary if k.startswith('layers_')],
        key=lambda x: int(x.split('_')[1])
    )
    if layer_variants:
        print(f"\nLayer Count Ablation:")
        print(f"  {'Layers':>6}  {'Test MSE':>12}  {'Test MAE':>12}  {'Params':>10}")
        for v in layer_variants:
            d = summary[v]
            n = int(v.split('_')[1])
            print(f"  {n:>6}  {d['test_mse_mean']:>12.6f}  "
                  f"{d['test_mae_mean']:>12.6f}  {d['n_params']:>10,}")

    # Hidden dim
    hidden_variants = sorted(
        [k for k in summary if k.startswith('hidden_')],
        key=lambda x: int(x.split('_')[1])
    )
    if hidden_variants:
        print(f"\nHidden Dimension Ablation:")
        print(f"  {'Hidden':>6}  {'Test MSE':>12}  {'Test MAE':>12}  {'Params':>10}")
        for v in hidden_variants:
            d = summary[v]
            h = int(v.split('_')[1])
            print(f"  {h:>6}  {d['test_mse_mean']:>12.6f}  "
                  f"{d['test_mae_mean']:>12.6f}  {d['n_params']:>10,}")


def main():
    parser = argparse.ArgumentParser(description='Exp7: Ablation Study')
    parser.add_argument('--molecule', type=str, default='ethanol')
    parser.add_argument('--n-train', type=int, default=1000)
    parser.add_argument('--n-val', type=int, default=200)
    parser.add_argument('--n-test', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-dir', type=str, default='./results/ablation')
    args = parser.parse_args()

    config = AblationConfig(
        molecule=args.molecule,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        seeds=args.seeds,
        device=args.device,
        save_dir=args.save_dir,
    )

    run_full_ablation(config)


if __name__ == '__main__':
    main()
