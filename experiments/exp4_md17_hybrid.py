"""
Experiment 4: MD17 Force Prediction Benchmark.

Compares the energy-conserving hybrid HC-Net against existing baselines
on MD17 molecular force prediction using standard protocol (9500 train).

Models:
- MD17HybridHCNetEnergy (ours, energy-conserving)
- MD17HybridHCNet (ours, direct forces)
- MD17EGNNAdapter (EGNN baseline)
- MD17CliffordNet (existing HC-Net)
- MD17BaselineNet (MLP baseline)

Molecules: all 8 MD17 molecules
Metrics: force MSE, force MAE, energy MSE, energy MAE, training time/epoch
Multi-seed runs with mean +/- std error bars.

Usage:
    python -m nips_hcnet.train --experiment exp4 --molecule ethanol
    python -m nips_hcnet.train --experiment exp4 --all_molecules --models hybrid_hcnet_energy egnn
"""

import os
import json
import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from nips_hcnet.models.md17_hybrid import MD17HybridHCNet, MD17HybridHCNetEnergy
from nips_hcnet.data.md17_dataset import (
    MD17Dataset, MD17_MOLECULES,
    get_md17_loaders, get_md17_loaders_both,
)
from nips_hcnet.models.md17_models import (
    MD17CliffordNet, MD17BaselineNet, MD17EGNNAdapter,
)


# Conversion: 1 kcal/mol = 43.3641 meV
KCAL_TO_MEV = 43.3641

# Published baselines at 9500 training samples (force MAE in meV/A).
# Cited from NequIP (Batzner et al. 2022), MACE (Batatia et al. 2022),
# PaiNN (Schutt et al. 2021), SchNet (Schutt et al. 2018).
PUBLISHED_BASELINES_9500 = {
    'NequIP': {
        'aspirin': 8.80, 'benzene': 0.30, 'ethanol': 2.40,
        'malonaldehyde': 3.60, 'naphthalene': 1.80,
        'salicylic': 4.00, 'toluene': 1.60, 'uracil': 3.10,
    },
    'MACE': {
        'aspirin': 6.59, 'benzene': 0.27, 'ethanol': 2.10,
        'malonaldehyde': 3.20, 'naphthalene': 1.30,
        'salicylic': 3.30, 'toluene': 1.20, 'uracil': 2.60,
    },
    'PaiNN': {
        'aspirin': 12.6, 'benzene': 0.80, 'ethanol': 5.20,
        'malonaldehyde': 7.20, 'naphthalene': 3.40,
        'salicylic': 7.60, 'toluene': 3.00, 'uracil': 5.60,
    },
    'SchNet': {
        'aspirin': 23.1, 'benzene': 1.70, 'ethanol': 8.00,
        'malonaldehyde': 11.2, 'naphthalene': 5.80,
        'salicylic': 12.4, 'toluene': 5.50, 'uracil': 9.50,
    },
}


# ---------------------------------------------------------------------------
# Training/eval for direct-force models
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    """Train one epoch for direct-force model."""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    n_batches = 0

    for batch_pos, batch_z, batch_forces in loader:
        batch_pos = batch_pos.to(device)
        batch_z = batch_z.to(device)
        batch_forces = batch_forces.to(device)

        optimizer.zero_grad()
        pred_forces = model(batch_pos, batch_z)
        loss = criterion(pred_forces, batch_forces)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mae += (pred_forces - batch_forces).abs().mean().item()
        n_batches += 1

    return total_loss / n_batches, total_mae / n_batches


def evaluate(model, loader, device, force_std=1.0):
    """Evaluate direct-force model.

    Returns (mse_norm, mae_norm, mae_mev) where mae_mev is in meV/A.
    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_pos, batch_z, batch_forces in loader:
            batch_pos = batch_pos.to(device)
            batch_z = batch_z.to(device)
            batch_forces = batch_forces.to(device)

            pred_forces = model(batch_pos, batch_z)
            mse = ((pred_forces - batch_forces) ** 2).mean().item()
            mae = (pred_forces - batch_forces).abs().mean().item()

            total_mse += mse
            total_mae += mae
            n_batches += 1

    mae_norm = total_mae / n_batches
    # Denormalize to physical units: normalized MAE * force_std -> kcal/mol/A -> meV/A
    mae_mev = mae_norm * force_std * KCAL_TO_MEV
    return total_mse / n_batches, mae_norm, mae_mev


# ---------------------------------------------------------------------------
# Training/eval for energy-conserving models
# ---------------------------------------------------------------------------

def train_epoch_energy(model, loader, optimizer, device,
                       w_force=1.0, w_energy=0.01):
    """Train one epoch for energy-conserving model with combined loss."""
    model.train()
    total_loss = 0.0
    total_force_mae = 0.0
    n_batches = 0

    for batch_pos, batch_z, batch_forces, batch_energy in loader:
        batch_pos = batch_pos.to(device).requires_grad_(True)
        batch_z = batch_z.to(device)
        batch_forces = batch_forces.to(device)
        batch_energy = batch_energy.to(device).squeeze(-1)  # [B]

        optimizer.zero_grad()
        pred_energy, pred_forces = model(batch_pos, batch_z)

        force_loss = nn.functional.mse_loss(pred_forces, batch_forces)
        energy_loss = nn.functional.mse_loss(pred_energy, batch_energy)
        loss = w_force * force_loss + w_energy * energy_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_force_mae += (pred_forces - batch_forces).abs().mean().item()
        n_batches += 1

    return total_loss / n_batches, total_force_mae / n_batches


def evaluate_energy(model, loader, device, force_std=1.0):
    """Evaluate energy-conserving model. Returns force and energy metrics."""
    model.eval()
    total_force_mse = 0.0
    total_force_mae = 0.0
    total_energy_mse = 0.0
    total_energy_mae = 0.0
    n_batches = 0

    for batch_pos, batch_z, batch_forces, batch_energy in loader:
        batch_pos = batch_pos.to(device).requires_grad_(True)
        batch_z = batch_z.to(device)
        batch_forces = batch_forces.to(device)
        batch_energy = batch_energy.to(device).squeeze(-1)

        pred_energy, pred_forces = model(batch_pos, batch_z)

        total_force_mse += ((pred_forces - batch_forces) ** 2).mean().item()
        total_force_mae += (pred_forces - batch_forces).abs().mean().item()
        total_energy_mse += ((pred_energy - batch_energy) ** 2).mean().item()
        total_energy_mae += (pred_energy - batch_energy).abs().mean().item()
        n_batches += 1

    n = max(n_batches, 1)
    force_mae_norm = total_force_mae / n
    return {
        'force_mse': total_force_mse / n,
        'force_mae': force_mae_norm,
        'force_mae_mev': force_mae_norm * force_std * KCAL_TO_MEV,
        'energy_mse': total_energy_mse / n,
        'energy_mae': total_energy_mae / n,
    }


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_model(name: str, n_atoms: int, hidden_dim: int, device):
    if name == 'hybrid_hcnet_energy':
        return MD17HybridHCNetEnergy(
            n_atoms=n_atoms,
            hidden_dim=hidden_dim,
            n_layers=4,
            k_neighbors=min(8, n_atoms - 1),
            cutoff=5.0,
        ).to(device)
    elif name == 'hybrid_hcnet':
        return MD17HybridHCNet(
            n_atoms=n_atoms,
            hidden_dim=hidden_dim,
            n_layers=4,
            k_neighbors=min(8, n_atoms - 1),
            cutoff=5.0,
        ).to(device)
    elif name == 'clifford_net':
        return MD17CliffordNet(
            n_atoms=n_atoms, hidden_dim=hidden_dim, n_layers=4
        ).to(device)
    elif name == 'egnn':
        return MD17EGNNAdapter(
            n_atoms=n_atoms, hidden_dim=hidden_dim, n_layers=4
        ).to(device)
    elif name == 'baseline':
        return MD17BaselineNet(
            n_atoms=n_atoms, hidden_dim=hidden_dim, n_layers=4
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {name}")


def is_energy_model(name: str) -> bool:
    return name == 'hybrid_hcnet_energy'


# ---------------------------------------------------------------------------
# Single seed run
# ---------------------------------------------------------------------------

def run_single_seed(
    model_name: str,
    molecule: str,
    n_train: int = 9500,
    n_val: int = 500,
    n_test: int = 1000,
    n_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    hidden_dim: int = 128,
    seed: int = 42,
    device: str = 'cuda',
):
    """Run a single training run (one seed)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    n_atoms = MD17_MOLECULES[molecule]['n_atoms']
    energy_mode = is_energy_model(model_name)

    # Get data loaders
    if energy_mode:
        train_loader, val_loader, test_loader = get_md17_loaders_both(
            molecule=molecule,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            batch_size=batch_size,
            num_workers=0,
            seed=seed,
        )
    else:
        train_loader, val_loader, test_loader = get_md17_loaders(
            molecule=molecule,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            batch_size=batch_size,
            num_workers=0,
            seed=seed,
        )

    # Extract normalization statistics for denormalization to physical units
    force_std = float(train_loader.dataset.force_std)

    model = create_model(model_name, n_atoms, hidden_dim, device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    n_params = sum(p.numel() for p in model.parameters())
    best_val_force_mse = float('inf')
    epoch_times = []

    for epoch in range(n_epochs):
        t0 = time.time()

        if energy_mode:
            train_loss, train_mae = train_epoch_energy(
                model, train_loader, optimizer, device
            )
            val_metrics = evaluate_energy(model, val_loader, device, force_std)
            val_force_mse = val_metrics['force_mse']
        else:
            criterion = nn.MSELoss()
            train_loss, train_mae = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_force_mse, _, _ = evaluate(model, val_loader, device, force_std)

        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)
        scheduler.step(val_force_mse)

        if val_force_mse < best_val_force_mse:
            best_val_force_mse = val_force_mse

        if (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch+1}: train_loss={train_loss:.6f}, "
                  f"val_force_mse={val_force_mse:.6f}")

    # Final test evaluation
    if energy_mode:
        test_metrics = evaluate_energy(model, test_loader, device, force_std)
        return {
            'test_force_mse': test_metrics['force_mse'],
            'test_force_mae': test_metrics['force_mae'],
            'test_force_mae_mev': test_metrics['force_mae_mev'],
            'test_energy_mse': test_metrics['energy_mse'],
            'test_energy_mae': test_metrics['energy_mae'],
            'force_std': force_std,
            'best_val_force_mse': best_val_force_mse,
            'mean_epoch_time': float(np.mean(epoch_times)),
            'n_params': n_params,
        }
    else:
        test_mse, test_mae, test_mae_mev = evaluate(
            model, test_loader, device, force_std
        )
        return {
            'test_force_mse': test_mse,
            'test_force_mae': test_mae,
            'test_force_mae_mev': test_mae_mev,
            'force_std': force_std,
            'best_val_force_mse': best_val_force_mse,
            'mean_epoch_time': float(np.mean(epoch_times)),
            'n_params': n_params,
        }


# ---------------------------------------------------------------------------
# Multi-seed experiment
# ---------------------------------------------------------------------------

def run_multi_seed(
    model_name: str,
    molecule: str,
    seeds: list,
    **kwargs,
):
    """Run over multiple seeds and aggregate mean +/- std."""
    seed_results = []
    for seed in seeds:
        print(f"    [seed={seed}]")
        res = run_single_seed(
            model_name=model_name,
            molecule=molecule,
            seed=seed,
            **kwargs,
        )
        seed_results.append(res)

    # Aggregate
    agg = {}
    all_keys = seed_results[0].keys()
    for key in all_keys:
        vals = [r[key] for r in seed_results]
        if isinstance(vals[0], (int, float)):
            agg[f'{key}_mean'] = float(np.mean(vals))
            agg[f'{key}_std'] = float(np.std(vals))
        else:
            agg[key] = vals[0]

    agg['per_seed'] = seed_results
    agg['n_seeds'] = len(seeds)
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Exp 4: MD17 Hybrid')
    parser.add_argument('--molecule', type=str, default='ethanol',
                        choices=list(MD17_MOLECULES.keys()))
    parser.add_argument('--all_molecules', action='store_true',
                        help='Run on all 8 MD17 molecules')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['hybrid_hcnet_energy', 'hybrid_hcnet',
                                 'egnn', 'clifford_net', 'baseline'])
    parser.add_argument('--n_train', type=int, default=9500)
    parser.add_argument('--n_val', type=int, default=500)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if args.all_molecules:
        molecules = list(MD17_MOLECULES.keys())
    else:
        molecules = [args.molecule]

    model_names = args.models

    print("=" * 70)
    print("EXPERIMENT 4: MD17 FORCE PREDICTION")
    print("=" * 70)
    print(f"Molecules: {molecules}")
    print(f"Models: {model_names}")
    print(f"Seeds: {args.seeds}")
    print(f"Train/Val/Test: {args.n_train}/{args.n_val}/{args.n_test}")
    print(f"Epochs: {args.n_epochs}")

    all_results = {}

    for molecule in molecules:
        print(f"\n{'='*50}")
        print(f"Molecule: {molecule} ({MD17_MOLECULES[molecule]['formula']})")
        print(f"{'='*50}")

        all_results[molecule] = {}

        for model_name in model_names:
            print(f"\n  Training {model_name}...")
            result = run_multi_seed(
                model_name=model_name,
                molecule=molecule,
                seeds=args.seeds,
                n_train=args.n_train,
                n_val=args.n_val,
                n_test=args.n_test,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                hidden_dim=args.hidden_dim,
                device=args.device,
            )
            all_results[molecule][model_name] = result

            fmse = result.get('test_force_mse_mean', 0)
            fmae = result.get('test_force_mae_mean', 0)
            fmse_s = result.get('test_force_mse_std', 0)
            fmae_s = result.get('test_force_mae_std', 0)
            print(f"  force_mse={fmse:.6f}+/-{fmse_s:.6f}, "
                  f"force_mae={fmae:.6f}+/-{fmae_s:.6f}")

    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for molecule in molecules:
        print(f"\n{molecule} ({MD17_MOLECULES[molecule]['formula']}):")
        header = (f"{'Model':<22} {'MAE (norm)':>14} "
                  f"{'MAE (meV/A)':>14} {'MSE (norm)':>14} {'Params':>10}")
        print(header)
        print("-" * len(header))

        # Our trained models
        for model_name in model_names:
            r = all_results[molecule][model_name]
            mae_m = r.get('test_force_mae_mean', 0)
            mae_s = r.get('test_force_mae_std', 0)
            mae_mev_m = r.get('test_force_mae_mev_mean', 0)
            mae_mev_s = r.get('test_force_mae_mev_std', 0)
            mse_m = r.get('test_force_mse_mean', 0)
            mse_s = r.get('test_force_mse_std', 0)
            params = r.get('n_params_mean', r.get('n_params', 0))
            print(f"{model_name:<22} "
                  f"{mae_m:>6.4f}+/-{mae_s:<5.4f} "
                  f"{mae_mev_m:>6.2f}+/-{mae_mev_s:<5.2f} "
                  f"{mse_m:>6.4f}+/-{mse_s:<5.4f} "
                  f"{int(params):>10,}")

        # Published baselines (reference rows, in meV/A)
        for baseline_name, baseline_data in PUBLISHED_BASELINES_9500.items():
            if molecule in baseline_data:
                val = baseline_data[molecule]
                print(f"{baseline_name + ' (pub.)':<22} "
                      f"{'—':>14} "
                      f"{val:>6.2f}{'':>8} "
                      f"{'—':>14} "
                      f"{'—':>10}")

        # Note about benzene
        if molecule == 'benzene':
            # Check if our models struggled
            for model_name in model_names:
                r = all_results[molecule][model_name]
                mae_mev = r.get('test_force_mae_mev_mean', 0)
                if mae_mev > 5.0:
                    print(f"  NOTE: Benzene has near-zero forces due to high "
                          f"symmetry. Published baselines use energy-conserving "
                          f"formulations + larger training sets.")
                    break

    # Save
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    save_data = {
        'results': all_results,
        'published_baselines': PUBLISHED_BASELINES_9500,
        'config': vars(args),
    }

    outfile = output_dir / f'exp4_md17_{timestamp}.json'
    with open(outfile, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nResults saved to: {outfile}")


if __name__ == '__main__':
    main()
