"""
Starved N-body Physics Experiment.

Tests Clifford vs Baseline with severely limited training data (N=100, 500)
to expose the geometric inductive bias advantage.

This is the "Figure 1" candidate per advisor recommendation:
"Run the Starved N-Body (N=100) experiment tonight. If you see a big gap there,
that is your Figure 1."

Hypothesis: With only 100-500 training samples, the Clifford model will maintain
performance due to geometric inductive bias, while baseline will struggle.

Usage:
    python -m pcnn.experiments.starved_nbody_exp
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from datetime import datetime
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from pcnn.data.nbody_dataset import get_nbody_loaders_with_ood
from pcnn.models.nbody_models import (
    CliffordNBodyNet,
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
    """Train N-body prediction model with detailed tracking."""
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
    epochs_to_threshold = None
    threshold = 0.01  # Track when we reach 0.01 MSE

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches

        # IID Test
        model.eval()
        test_loss = evaluate_model(model, test_loader, criterion, device)
        best_test_loss = min(best_test_loss, test_loss)

        # Track epochs to threshold
        if epochs_to_threshold is None and test_loss < threshold:
            epochs_to_threshold = epoch

        # OOD Test
        ood_losses = {}
        for angle, loader in ood_loaders.items():
            ood_losses[angle] = evaluate_model(model, loader, criterion, device, has_angle=(angle != 0))

        scheduler.step()

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
        'epochs_to_threshold': epochs_to_threshold,
        'final_ood_losses': {k: v[-1] for k, v in history['ood_losses'].items()},
        'history': history
    }


def evaluate_model(model, loader, criterion, device, has_angle=False):
    """Evaluate model on a data loader."""
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            if has_angle:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0


def run_starved_nbody_experiment(
    n_particles_list: List[int] = [3, 5, 7],
    n_train_list: List[int] = [100, 500],
    rotation_angles: List[float] = [0, 45, 90],
    seeds: List[int] = [42, 123, 456],
    epochs: int = 100,
    n_test: int = 1000,
    output_dir: str = './results/starved_nbody'
) -> Dict:
    """
    Run the Starved N-body experiment.

    Key difference from standard experiment: n_train = [100, 500] instead of 5000

    Args:
        n_particles_list: Number of particles to test
        n_train_list: Training sample counts (the "starvation" levels)
        rotation_angles: Rotation angles for OOD testing
        seeds: Random seeds
        epochs: Training epochs
        n_test: Test samples
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    print(f'\n*** STARVED N-BODY EXPERIMENT ***')
    print(f'Training samples: {n_train_list} (vs standard 5000)')
    print(f'This tests the "data desert" hypothesis\n')

    results = {
        'n_particles': n_particles_list,
        'n_train_list': n_train_list,
        'rotation_angles': rotation_angles,
        'seeds': seeds,
        'epochs': epochs,
        'clifford': {},
        'baseline': {},
        'timestamp': datetime.now().isoformat()
    }

    for n_train in n_train_list:
        print(f'\n{"#"*70}')
        print(f'TRAINING SAMPLES: {n_train}')
        print(f'{"#"*70}')

        results['clifford'][n_train] = {}
        results['baseline'][n_train] = {}

        for n_particles in n_particles_list:
            print(f'\n{"="*60}')
            print(f'N particles: {n_particles}, Training samples: {n_train}')
            print(f'{"="*60}')

            results['clifford'][n_train][n_particles] = []
            results['baseline'][n_train][n_particles] = []

            for seed in seeds:
                print(f'\n--- Seed {seed} ---')

                # Get data loaders
                train_loader, test_loader, ood_loaders = get_nbody_loaders_with_ood(
                    n_train=n_train,
                    n_test=n_test,
                    n_particles=n_particles,
                    rotation_angles=rotation_angles,
                    batch_size=min(64, n_train),  # Smaller batch for small data
                    seed=seed
                )
                print(f'Training samples: {len(train_loader.dataset)}')

                # Train Clifford
                print('Training Clifford model...')
                torch.manual_seed(seed)
                clifford = CliffordNBodyNet(n_particles=n_particles, hidden_dim=128)
                print(f'  Parameters: {count_parameters(clifford):,}')

                clifford_results = train_nbody_model(
                    clifford, train_loader, test_loader, ood_loaders,
                    epochs=epochs, device=device
                )
                results['clifford'][n_train][n_particles].append(clifford_results)
                print(f'  Final IID: {clifford_results["final_test_loss"]:.4f}')

                # Train Baseline
                print('Training Baseline model...')
                torch.manual_seed(seed)
                baseline = BaselineNBodyNetWithAttention(n_particles=n_particles, hidden_dim=128)
                print(f'  Parameters: {count_parameters(baseline):,}')

                baseline_results = train_nbody_model(
                    baseline, train_loader, test_loader, ood_loaders,
                    epochs=epochs, device=device
                )
                results['baseline'][n_train][n_particles].append(baseline_results)
                print(f'  Final IID: {baseline_results["final_test_loss"]:.4f}')

    # Compute summary
    summary = compute_summary(results)
    results['summary'] = summary

    # Save results
    save_results(results, output_dir)

    # Plot results
    plot_starved_comparison(results, output_dir)
    plot_sample_efficiency_physics(results, output_dir)

    # Print summary
    print_summary(results)

    print(f'\nResults saved to {output_dir}')
    return results


def compute_summary(results: Dict) -> Dict:
    """Compute summary statistics."""
    summary = {
        'n_train_list': results['n_train_list'],
        'n_particles': results['n_particles'],
        'rotation_angles': results['rotation_angles'],
        'by_n_train': {}
    }

    for n_train in results['n_train_list']:
        summary['by_n_train'][n_train] = {'clifford': {}, 'baseline': {}}

        for n_particles in results['n_particles']:
            # Clifford
            clifford_iid = [r['final_test_loss'] for r in results['clifford'][n_train][n_particles]]
            summary['by_n_train'][n_train]['clifford'][n_particles] = {
                'iid_mean': np.mean(clifford_iid),
                'iid_std': np.std(clifford_iid),
                'ood': {}
            }
            for angle in results['rotation_angles']:
                ood = [r['final_ood_losses'][angle] for r in results['clifford'][n_train][n_particles]]
                summary['by_n_train'][n_train]['clifford'][n_particles]['ood'][angle] = {
                    'mean': np.mean(ood),
                    'std': np.std(ood)
                }

            # Baseline
            baseline_iid = [r['final_test_loss'] for r in results['baseline'][n_train][n_particles]]
            summary['by_n_train'][n_train]['baseline'][n_particles] = {
                'iid_mean': np.mean(baseline_iid),
                'iid_std': np.std(baseline_iid),
                'ood': {}
            }
            for angle in results['rotation_angles']:
                ood = [r['final_ood_losses'][angle] for r in results['baseline'][n_train][n_particles]]
                summary['by_n_train'][n_train]['baseline'][n_particles]['ood'][angle] = {
                    'mean': np.mean(ood),
                    'std': np.std(ood)
                }

    return summary


def save_results(results: Dict, output_dir: str):
    """Save results to JSON."""
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

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)


def plot_starved_comparison(results: Dict, output_dir: str):
    """Plot Clifford vs Baseline at different starvation levels."""
    n_train_list = results['n_train_list']
    n_particles = results['n_particles']
    angles = results['rotation_angles']

    fig, axes = plt.subplots(len(n_train_list), len(n_particles),
                              figsize=(5*len(n_particles), 4*len(n_train_list)))

    if len(n_train_list) == 1:
        axes = [axes]

    for row_idx, n_train in enumerate(n_train_list):
        for col_idx, n_part in enumerate(n_particles):
            ax = axes[row_idx][col_idx] if len(n_particles) > 1 else axes[row_idx]

            # Get data
            clifford_means = []
            clifford_stds = []
            baseline_means = []
            baseline_stds = []

            for angle in angles:
                c_losses = [r['final_ood_losses'][angle] for r in results['clifford'][n_train][n_part]]
                b_losses = [r['final_ood_losses'][angle] for r in results['baseline'][n_train][n_part]]

                clifford_means.append(np.mean(c_losses))
                clifford_stds.append(np.std(c_losses))
                baseline_means.append(np.mean(b_losses))
                baseline_stds.append(np.std(b_losses))

            ax.errorbar(angles, clifford_means, yerr=clifford_stds,
                       marker='o', label='Clifford', capsize=5, linewidth=2)
            ax.errorbar(angles, baseline_means, yerr=baseline_stds,
                       marker='s', label='Baseline', capsize=5, linewidth=2)

            ax.set_xlabel('Rotation Angle (°)')
            ax.set_ylabel('MSE Loss')
            ax.set_title(f'N={n_part} particles, Train={n_train}')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.suptitle('Starved N-Body: OOD Generalization with Limited Data', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'starved_ood_comparison.png'), dpi=150)
    plt.close()


def plot_sample_efficiency_physics(results: Dict, output_dir: str):
    """Plot how performance scales with training data."""
    n_train_list = results['n_train_list']
    n_particles = results['n_particles']

    # Use middle particle count
    n_part = n_particles[len(n_particles) // 2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # IID performance
    ax = axes[0]
    clifford_iid = []
    baseline_iid = []

    for n_train in n_train_list:
        c_losses = [r['final_test_loss'] for r in results['clifford'][n_train][n_part]]
        b_losses = [r['final_test_loss'] for r in results['baseline'][n_train][n_part]]
        clifford_iid.append((np.mean(c_losses), np.std(c_losses)))
        baseline_iid.append((np.mean(b_losses), np.std(b_losses)))

    ax.errorbar(n_train_list, [x[0] for x in clifford_iid], yerr=[x[1] for x in clifford_iid],
               marker='o', label='Clifford', capsize=5, linewidth=2, markersize=10)
    ax.errorbar(n_train_list, [x[0] for x in baseline_iid], yerr=[x[1] for x in baseline_iid],
               marker='s', label='Baseline', capsize=5, linewidth=2, markersize=10)

    ax.set_xlabel('Training Samples')
    ax.set_ylabel('IID Test MSE')
    ax.set_title(f'Sample Efficiency (N={n_part} particles)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # Improvement ratio
    ax = axes[1]
    ratios_by_angle = {angle: [] for angle in results['rotation_angles']}

    for n_train in n_train_list:
        for angle in results['rotation_angles']:
            c_losses = [r['final_ood_losses'][angle] for r in results['clifford'][n_train][n_part]]
            b_losses = [r['final_ood_losses'][angle] for r in results['baseline'][n_train][n_part]]
            ratio = np.mean(c_losses) / np.mean(b_losses) if np.mean(b_losses) > 0 else 1
            ratios_by_angle[angle].append(ratio)

    for angle in results['rotation_angles']:
        ax.plot(n_train_list, ratios_by_angle[angle], 'o-', label=f'{angle}°', linewidth=2, markersize=8)

    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('Clifford/Baseline MSE Ratio')
    ax.set_title('Advantage Ratio (< 1.0 = Clifford better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_efficiency_physics.png'), dpi=150)
    plt.close()


def print_summary(results: Dict):
    """Print summary table."""
    print('\n' + '='*80)
    print('SUMMARY: STARVED N-BODY EXPERIMENT')
    print('='*80)

    for n_train in results['n_train_list']:
        print(f'\n*** Training Samples: {n_train} ***')

        for n_part in results['n_particles']:
            print(f'\n  N = {n_part} particles:')
            print(f'  {"-"*60}')
            print(f'  {"Angle":>10} {"Clifford MSE":>18} {"Baseline MSE":>18} {"Ratio":>10}')
            print(f'  {"-"*60}')

            for angle in results['rotation_angles']:
                c_losses = [r['final_ood_losses'][angle] for r in results['clifford'][n_train][n_part]]
                b_losses = [r['final_ood_losses'][angle] for r in results['baseline'][n_train][n_part]]

                c_mean, c_std = np.mean(c_losses), np.std(c_losses)
                b_mean, b_std = np.mean(b_losses), np.std(b_losses)
                ratio = c_mean / b_mean if b_mean > 0 else 0

                c_str = f'{c_mean:.4f} +/- {c_std:.4f}'
                b_str = f'{b_mean:.4f} +/- {b_std:.4f}'

                marker = '***' if ratio < 0.8 else ('**' if ratio < 0.9 else ('*' if ratio < 1.0 else ''))
                print(f'  {angle:>10}° {c_str:>18} {b_str:>18} {ratio:>8.2f} {marker}')

    print('\n' + '='*80)
    print('*** = Clifford 20%+ better, ** = 10%+ better, * = marginally better')
    print('Ratio < 1.0 means Clifford outperforms Baseline')
    print('='*80)


def main():
    parser = argparse.ArgumentParser(description='Starved N-body Experiment')
    parser.add_argument('--particles', type=int, nargs='+', default=[3, 5, 7])
    parser.add_argument('--n-train', type=int, nargs='+', default=[100, 500])
    parser.add_argument('--angles', type=float, nargs='+', default=[0, 45, 90])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--output', type=str, default='./results/starved_nbody')
    args = parser.parse_args()

    run_starved_nbody_experiment(
        n_particles_list=args.particles,
        n_train_list=args.n_train,
        rotation_angles=args.angles,
        seeds=args.seeds,
        epochs=args.epochs,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
