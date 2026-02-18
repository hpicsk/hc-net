"""
Spinning System Experiment: Vector Averaging Collapse Demonstration.

This is the "killer experiment" that proves HC-Net's novelty:
- VectorMeanFieldNet FAILS on spinning systems (velocity vectors cancel)
- HC-Net SUCCEEDS because bivectors preserve angular momentum

Hypothesis:
- In spinning systems, sum(v_i) ≈ 0 (vectors cancel due to symmetry)
- But sum(r_i ∧ v_i) ≠ 0 (bivectors = angular momentum, same sign)
- VectorMeanFieldNet's global mean is uninformative (~0)
- HC-Net's multivector mean preserves rotation via bivector components

Expected Results:
- HC-Net: Low MSE on spinning systems
- VectorMeanFieldNet: High MSE (fails to capture rotation)
- HCNetNoBivector: Intermediate (has mean-field but no bivectors)
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.nbody_models import CliffordNBodyNet, CliffordNBodyNetNoBivector
from models.vector_meanfield import VectorMeanFieldNet
from data.spinning_nbody import SpinningNBodyDataset, get_spinning_nbody_loaders


def create_model(model_name: str, n_particles: int = 5, hidden_dim: int = 128) -> nn.Module:
    """Create model by name."""
    if model_name == 'hcnet':
        return CliffordNBodyNet(
            n_particles=n_particles,
            hidden_dim=hidden_dim,
            n_layers=4,
            use_attention=False,
            use_mean_field=True
        )
    elif model_name == 'hcnet_no_bivector':
        return CliffordNBodyNetNoBivector(
            n_particles=n_particles,
            hidden_dim=hidden_dim,
            n_layers=4,
            use_mean_field=True
        )
    elif model_name == 'vector_meanfield':
        return VectorMeanFieldNet(
            n_particles=n_particles,
            hidden_dim=hidden_dim,
            n_layers=4,
            use_mean_field=True
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_inp, batch_tgt in loader:
        batch_inp = batch_inp.to(device)
        batch_tgt = batch_tgt.to(device)

        optimizer.zero_grad()
        pred = model(batch_inp)
        loss = criterion(pred, batch_tgt)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> float:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_inp, batch_tgt in loader:
            batch_inp = batch_inp.to(device)
            batch_tgt = batch_tgt.to(device)

            pred = model(batch_inp)
            loss = criterion(pred, batch_tgt)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def run_experiment(
    model_name: str,
    n_train: int = 5000,
    n_test: int = 1000,
    n_particles: int = 5,
    hidden_dim: int = 128,
    n_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 0.001,
    seed: int = 42,
    device: str = 'cuda'
) -> dict:
    """Run single experiment."""
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Data
    train_loader, test_loader, stats = get_spinning_nbody_loaders(
        n_train=n_train,
        n_test=n_test,
        n_particles=n_particles,
        batch_size=batch_size,
        num_workers=4,
        seed=seed
    )

    # Model
    model = create_model(model_name, n_particles, hidden_dim)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Training loop
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []

    start_time = time.time()

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: train={train_loss:.6f}, test={test_loss:.6f}")

    training_time = time.time() - start_time

    return {
        'model': model_name,
        'n_params': n_params,
        'best_test_loss': best_test_loss,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
        'train_losses': train_losses,
        'test_losses': test_losses,
        'training_time': training_time,
        'dataset_stats': stats,
        'config': {
            'n_train': n_train,
            'n_test': n_test,
            'n_particles': n_particles,
            'hidden_dim': hidden_dim,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'seed': seed
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Spinning System Experiment')
    parser.add_argument('--n_train', type=int, default=5000, help='Training samples')
    parser.add_argument('--n_test', type=int, default=1000, help='Test samples')
    parser.add_argument('--n_particles', type=int, default=5, help='Particles per system')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--n_epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                        help='Random seeds')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output_dir', type=str, default='results/spinning_system',
                        help='Output directory')
    args = parser.parse_args()

    # Models to compare
    models = ['hcnet', 'vector_meanfield', 'hcnet_no_bivector']

    # Create output directory
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results storage
    all_results = {model: [] for model in models}

    print("=" * 70)
    print("SPINNING SYSTEM EXPERIMENT: Vector Averaging Collapse Demonstration")
    print("=" * 70)
    print(f"\nHypothesis: VectorMeanFieldNet will FAIL because sum(v_i) ≈ 0")
    print(f"           HC-Net will SUCCEED because sum(B_ij) ≠ 0 (angular momentum)")
    print(f"\nConfig:")
    print(f"  Training samples: {args.n_train}")
    print(f"  Test samples: {args.n_test}")
    print(f"  Particles: {args.n_particles}")
    print(f"  Epochs: {args.n_epochs}")
    print(f"  Seeds: {args.seeds}")
    print("=" * 70)

    for seed in args.seeds:
        print(f"\n{'='*70}")
        print(f"SEED: {seed}")
        print(f"{'='*70}")

        for model_name in models:
            print(f"\n--- Training {model_name} ---")

            result = run_experiment(
                model_name=model_name,
                n_train=args.n_train,
                n_test=args.n_test,
                n_particles=args.n_particles,
                hidden_dim=args.hidden_dim,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=seed,
                device=args.device
            )

            all_results[model_name].append(result)
            print(f"  Best test MSE: {result['best_test_loss']:.6f}")
            print(f"  Training time: {result['training_time']:.1f}s")

    # Aggregate results
    print("\n" + "=" * 70)
    print("FINAL RESULTS (mean ± std across seeds)")
    print("=" * 70)

    summary = {}
    for model_name in models:
        test_losses = [r['best_test_loss'] for r in all_results[model_name]]
        mean_loss = np.mean(test_losses)
        std_loss = np.std(test_losses)
        summary[model_name] = {
            'mean_mse': mean_loss,
            'std_mse': std_loss,
            'all_mse': test_losses
        }
        print(f"{model_name:20s}: MSE = {mean_loss:.6f} ± {std_loss:.6f}")

    # Compute ratios
    hcnet_mse = summary['hcnet']['mean_mse']
    vector_mse = summary['vector_meanfield']['mean_mse']
    ratio = vector_mse / hcnet_mse

    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    print(f"VectorMeanFieldNet / HC-Net MSE ratio: {ratio:.2f}x")

    if ratio > 2.0:
        print(f"\n✓ HYPOTHESIS CONFIRMED: VectorMeanFieldNet fails on spinning systems")
        print(f"  Vector averaging collapses rotational information (sum(v_i) ≈ 0)")
        print(f"  HC-Net preserves it via bivector mean-field (sum(B_ij) ≠ 0)")
    else:
        print(f"\n? UNEXPECTED: Ratio is {ratio:.2f}x (expected > 2x)")
        print(f"  This may indicate the dataset doesn't have enough rotation symmetry")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'spinning_system_{timestamp}.json'

    # Convert numpy arrays for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj

    save_data = {
        'summary': summary,
        'all_results': convert_for_json(all_results),
        'config': vars(args),
        'timestamp': timestamp,
        'conclusion': {
            'vector_hcnet_ratio': float(ratio),
            'hypothesis_confirmed': bool(ratio > 2.0)
        }
    }

    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return summary


if __name__ == '__main__':
    main()
