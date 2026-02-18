"""
2D N-body Physics Prediction Experiment.

Tests whether Clifford-based models generalize to rotated systems better
than baseline due to native geometric representation of vectors.

Hypothesis: The Clifford model's geometric product creates bivector terms
that naturally encode rotation-invariant relationships between position
and velocity vectors. This should lead to better generalization when
test systems are rotated compared to training data.

Usage:
    python -m pcnn.experiments.nbody_physics_exp
    python -m pcnn.experiments.nbody_physics_exp --particles 3 5 7 --angles 0 45 90
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from datetime import datetime
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from pcnn.data.nbody_dataset import get_nbody_loaders_with_ood
from pcnn.models.nbody_models import (
    CliffordNBodyNet,
    BaselineNBodyNet,
    BaselineNBodyNetWithAttention,
    count_parameters
)


def train_nbody_model(
    model: nn.Module,
    train_loader,
    test_loader,
    ood_loaders: Dict[float, object],
    epochs: int = 100,
    device: str = 'cuda',
    lr: float = 1e-3,
    verbose: bool = True
) -> Dict:
    """
    Train N-body prediction model.

    Args:
        model: Model to train
        train_loader: Training data
        test_loader: IID test data
        ood_loaders: Dict mapping rotation angle to DataLoader
        epochs: Number of training epochs
        device: Device to use
        lr: Learning rate
        verbose: Whether to print progress

    Returns:
        Results dictionary with IID and OOD metrics
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'train_loss': [],
        'test_loss': [],
        'ood_losses': {angle: [] for angle in ood_loaders.keys()}
    }

    best_test_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        n_batches = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches

        # IID Test
        model.eval()
        test_loss = evaluate_model(model, test_loader, criterion, device)
        best_test_loss = min(best_test_loss, test_loss)

        # OOD Test (rotated systems)
        ood_losses = {}
        for angle, loader in ood_loaders.items():
            ood_losses[angle] = evaluate_model(model, loader, criterion, device, has_angle=(angle != 0))

        scheduler.step()

        # Log
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        for angle, loss in ood_losses.items():
            history['ood_losses'][angle].append(loss)

        if verbose and epoch % 20 == 0:
            ood_str = ', '.join([f'{a}°:{l:.4f}' for a, l in sorted(ood_losses.items())])
            print(f'  Epoch {epoch}: Train {train_loss:.4f}, Test {test_loss:.4f}, OOD [{ood_str}]')

    return {
        'final_train_loss': history['train_loss'][-1],
        'final_test_loss': history['test_loss'][-1],
        'best_test_loss': best_test_loss,
        'final_ood_losses': {k: v[-1] for k, v in history['ood_losses'].items()},
        'history': history
    }


def evaluate_model(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
    has_angle: bool = False
) -> float:
    """Evaluate model on a data loader."""
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            if has_angle:
                inputs, targets, _ = batch  # Rotated dataset includes angle
            else:
                inputs, targets = batch

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0


def run_nbody_experiment(
    n_particles_list: List[int] = [3, 5, 7],
    rotation_angles: List[float] = [0, 45, 90],
    seeds: List[int] = [42, 123, 456],
    epochs: int = 100,
    n_train: int = 5000,
    n_test: int = 1000,
    output_dir: str = './results/nbody_physics'
) -> Dict:
    """
    Run the N-body physics experiment.

    Key tests:
    1. IID generalization (same distribution as training)
    2. OOD generalization to rotated systems

    Hypothesis: Clifford model with geometric operations generalizes
    better to rotated systems due to rotation-equivariant representations.

    Args:
        n_particles_list: Number of particles to test
        rotation_angles: Rotation angles for OOD testing
        seeds: Random seeds for multiple runs
        epochs: Training epochs
        n_train: Training samples
        n_test: Test samples
        output_dir: Output directory

    Returns:
        Results dictionary
    """
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    results = {
        'n_particles': n_particles_list,
        'rotation_angles': rotation_angles,
        'seeds': seeds,
        'epochs': epochs,
        'clifford': {},
        'baseline': {},
        'timestamp': datetime.now().isoformat()
    }

    for n_particles in n_particles_list:
        print(f'\n{"="*60}')
        print(f'N particles: {n_particles}')
        print(f'{"="*60}')

        results['clifford'][n_particles] = []
        results['baseline'][n_particles] = []

        for seed in seeds:
            print(f'\n--- Seed {seed} ---')

            # Get data loaders
            train_loader, test_loader, ood_loaders = get_nbody_loaders_with_ood(
                n_train=n_train,
                n_test=n_test,
                n_particles=n_particles,
                rotation_angles=rotation_angles,
                batch_size=128,
                seed=seed
            )
            print(f'Training samples: {len(train_loader.dataset)}')

            # Train Clifford model
            print('Training Clifford model...')
            torch.manual_seed(seed)
            clifford = CliffordNBodyNet(n_particles=n_particles, hidden_dim=128)
            print(f'  Parameters: {count_parameters(clifford):,}')

            clifford_results = train_nbody_model(
                clifford, train_loader, test_loader, ood_loaders,
                epochs=epochs, device=device
            )
            results['clifford'][n_particles].append(clifford_results)
            print(f'  Final IID: {clifford_results["final_test_loss"]:.4f}')

            # Train Baseline model (with attention for fair comparison)
            print('Training Baseline model...')
            torch.manual_seed(seed)
            baseline = BaselineNBodyNetWithAttention(n_particles=n_particles, hidden_dim=128)
            print(f'  Parameters: {count_parameters(baseline):,}')

            baseline_results = train_nbody_model(
                baseline, train_loader, test_loader, ood_loaders,
                epochs=epochs, device=device
            )
            results['baseline'][n_particles].append(baseline_results)
            print(f'  Final IID: {baseline_results["final_test_loss"]:.4f}')

    # Compute summary
    summary = compute_ood_summary(results)
    results['summary'] = summary

    # Save results
    save_results(results, output_dir)

    # Plot results
    plot_ood_generalization(summary, output_dir)
    plot_rotation_degradation(results, output_dir)

    # Print summary
    print_summary(summary, rotation_angles)

    print(f'\nResults saved to {output_dir}')
    return results


def compute_ood_summary(results: Dict) -> Dict:
    """Compute summary statistics from results."""
    summary = {
        'n_particles': results['n_particles'],
        'rotation_angles': results['rotation_angles'],
        'clifford': {'iid': [], 'ood': {a: [] for a in results['rotation_angles']}},
        'baseline': {'iid': [], 'ood': {a: [] for a in results['rotation_angles']}}
    }

    for n in results['n_particles']:
        # Clifford
        clifford_iid = [r['final_test_loss'] for r in results['clifford'][n]]
        summary['clifford']['iid'].append({
            'mean': np.mean(clifford_iid),
            'std': np.std(clifford_iid)
        })

        for angle in results['rotation_angles']:
            clifford_ood = [r['final_ood_losses'][angle] for r in results['clifford'][n]]
            summary['clifford']['ood'][angle].append({
                'mean': np.mean(clifford_ood),
                'std': np.std(clifford_ood)
            })

        # Baseline
        baseline_iid = [r['final_test_loss'] for r in results['baseline'][n]]
        summary['baseline']['iid'].append({
            'mean': np.mean(baseline_iid),
            'std': np.std(baseline_iid)
        })

        for angle in results['rotation_angles']:
            baseline_ood = [r['final_ood_losses'][angle] for r in results['baseline'][n]]
            summary['baseline']['ood'][angle].append({
                'mean': np.mean(baseline_ood),
                'std': np.std(baseline_ood)
            })

    return summary


def save_results(results: Dict, output_dir: str):
    """Save results to JSON file."""
    results_path = os.path.join(output_dir, 'results.json')

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {str(k): convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    with open(results_path, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)


def plot_ood_generalization(summary: Dict, output_dir: str):
    """
    Plot OOD generalization results.

    Shows how prediction error changes with rotation angle
    for Clifford vs Baseline.
    """
    angles = summary['rotation_angles']
    n_particles = summary['n_particles']

    fig, axes = plt.subplots(1, len(n_particles), figsize=(5*len(n_particles), 5))
    if len(n_particles) == 1:
        axes = [axes]

    for ax, n in zip(axes, n_particles):
        idx = n_particles.index(n)

        # Clifford
        clifford_means = [summary['clifford']['ood'][a][idx]['mean'] for a in angles]
        clifford_stds = [summary['clifford']['ood'][a][idx]['std'] for a in angles]

        # Baseline
        baseline_means = [summary['baseline']['ood'][a][idx]['mean'] for a in angles]
        baseline_stds = [summary['baseline']['ood'][a][idx]['std'] for a in angles]

        ax.errorbar(angles, clifford_means, yerr=clifford_stds,
                    marker='o', label='Clifford', capsize=5, linewidth=2, markersize=8)
        ax.errorbar(angles, baseline_means, yerr=baseline_stds,
                    marker='s', label='Baseline', capsize=5, linewidth=2, markersize=8)

        ax.set_xlabel('Rotation Angle (degrees)')
        ax.set_ylabel('MSE Loss')
        ax.set_title(f'N = {n} particles')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('OOD Generalization: Rotated N-body Systems', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ood_generalization.png'), dpi=150)
    plt.close()


def plot_rotation_degradation(results: Dict, output_dir: str):
    """
    Plot how much performance degrades with rotation.

    Computes: (OOD_loss - IID_loss) / IID_loss as percentage
    """
    angles = results['rotation_angles']
    n_particles = results['n_particles']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use middle n_particles for cleaner plot
    n = n_particles[len(n_particles) // 2]
    idx = n_particles.index(n)

    # Get IID baseline
    clifford_iid = np.mean([r['final_test_loss'] for r in results['clifford'][n]])
    baseline_iid = np.mean([r['final_test_loss'] for r in results['baseline'][n]])

    # Compute degradation
    clifford_degradation = []
    baseline_degradation = []

    for angle in angles:
        clifford_ood = np.mean([r['final_ood_losses'][angle] for r in results['clifford'][n]])
        baseline_ood = np.mean([r['final_ood_losses'][angle] for r in results['baseline'][n]])

        clifford_degradation.append(100 * (clifford_ood - clifford_iid) / clifford_iid)
        baseline_degradation.append(100 * (baseline_ood - baseline_iid) / baseline_iid)

    ax.plot(angles, clifford_degradation, 'o-', label='Clifford', linewidth=2, markersize=8)
    ax.plot(angles, baseline_degradation, 's-', label='Baseline', linewidth=2, markersize=8)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Rotation Angle (degrees)')
    ax.set_ylabel('Performance Degradation (%)')
    ax.set_title(f'Rotation Sensitivity (N={n} particles)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rotation_degradation.png'), dpi=150)
    plt.close()


def print_summary(summary: Dict, angles: List[float]):
    """Print summary table."""
    print('\n' + '='*80)
    print('SUMMARY: N-body Physics OOD Generalization')
    print('='*80)

    for n_idx, n in enumerate(summary['n_particles']):
        print(f'\nN = {n} particles:')
        print('-'*60)
        print(f'{"Angle":>10} {"Clifford MSE":>20} {"Baseline MSE":>20} {"Ratio":>10}')
        print('-'*60)

        for angle in angles:
            c_mean = summary['clifford']['ood'][angle][n_idx]['mean']
            c_std = summary['clifford']['ood'][angle][n_idx]['std']
            b_mean = summary['baseline']['ood'][angle][n_idx]['mean']
            b_std = summary['baseline']['ood'][angle][n_idx]['std']

            c_str = f'{c_mean:.4f} +/- {c_std:.4f}'
            b_str = f'{b_mean:.4f} +/- {b_std:.4f}'
            ratio = c_mean / b_mean if b_mean > 0 else 0

            print(f'{angle:>10}° {c_str:>20} {b_str:>20} {ratio:>10.2f}')

    print('='*80)
    print('\nRatio < 1.0 means Clifford outperforms Baseline')
    print('Win condition: Clifford ratio stays constant while Baseline degrades at higher angles')


def main():
    parser = argparse.ArgumentParser(description='N-body Physics Experiment')
    parser.add_argument('--particles', type=int, nargs='+', default=[3, 5, 7],
                        help='Number of particles to test')
    parser.add_argument('--angles', type=float, nargs='+', default=[0, 45, 90],
                        help='Rotation angles for OOD testing')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                        help='Random seeds')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--n-train', type=int, default=5000,
                        help='Number of training samples')
    parser.add_argument('--n-test', type=int, default=1000,
                        help='Number of test samples')
    parser.add_argument('--output', type=str, default='./results/nbody_physics',
                        help='Output directory')
    args = parser.parse_args()

    run_nbody_experiment(
        n_particles_list=args.particles,
        rotation_angles=args.angles,
        seeds=args.seeds,
        epochs=args.epochs,
        n_train=args.n_train,
        n_test=args.n_test,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
