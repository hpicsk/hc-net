"""
Relational MNIST Sample Efficiency Experiment.

Tests HC-Net vs Baseline on the harder 3-way binding task:
"Red Triangle LEFT-OF Blue Circle"

This task requires:
1. Color binding (Red, Blue)
2. Shape binding (Triangle, Circle)
3. Spatial relation (LEFT-OF)

Hypothesis: HC-Net solves this with N=100 samples while Baseline
requires N=2000+ due to the need for coordinate-based logic.

Usage:
    python -m pcnn.experiments.relational_mnist_exp
    python -m pcnn.experiments.relational_mnist_exp --samples 10 50 100 500 1000 2000
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

from pcnn.data.relational_mnist import get_relational_mnist_loaders
from pcnn.models import HCNetResNetSmall
from pcnn.models.baseline_resnet import BaselineResNetSmall


def train_model(
    model: nn.Module,
    train_loader,
    test_loader,
    epochs: int = 50,
    device: str = 'cuda',
    lr: float = 0.01,
    verbose: bool = True
) -> Dict:
    """Train model and return results."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}
    best_test_acc = 0
    epochs_to_90 = None

    for epoch in range(epochs):
        # Train
        model.train()
        correct = 0
        total = 0
        train_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        train_loss = train_loss / len(train_loader)

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        test_loss = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100. * correct / total
        test_loss = test_loss / len(test_loader)
        best_test_acc = max(best_test_acc, test_acc)

        # Track epochs to 90%
        if epochs_to_90 is None and test_acc >= 90:
            epochs_to_90 = epoch

        scheduler.step()

        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)

        if verbose and epoch % 10 == 0:
            print(f'  Epoch {epoch}: Train {train_acc:.1f}%, Test {test_acc:.1f}%')

    return {
        'best_test_acc': best_test_acc,
        'final_test_acc': history['test_acc'][-1],
        'best_train_acc': max(history['train_acc']),
        'generalization_gap': max(history['train_acc']) - best_test_acc,
        'epochs_to_90': epochs_to_90,
        'history': history
    }


def run_relational_mnist_experiment(
    samples_per_class: List[int] = [10, 50, 100, 500, 1000, 2000],
    seeds: List[int] = [42, 123, 456],
    epochs: int = 50,
    output_dir: str = './results/relational_mnist',
    n_distractors: int = 0
) -> Dict:
    """
    Run the Relational MNIST experiment.

    Tests 3-way compositional binding: Color × Shape × Position.

    Args:
        samples_per_class: Training samples per class
        seeds: Random seeds
        epochs: Training epochs
        output_dir: Output directory
        n_distractors: Number of distractor objects (makes task harder)

    Returns:
        Results dictionary
    """
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    print(f'\n*** RELATIONAL MNIST EXPERIMENT ***')
    print(f'Task: "Red Triangle LEFT-OF Blue Circle"')
    if n_distractors > 0:
        print(f'With {n_distractors} distractor objects ({2 + n_distractors} total)')
    print(f'This tests 3-way binding (Color × Shape × Position)\n')

    results = {
        'samples': samples_per_class,
        'seeds': seeds,
        'epochs': epochs,
        'n_distractors': n_distractors,
        'hcnet': {n: [] for n in samples_per_class},
        'baseline': {n: [] for n in samples_per_class},
        'timestamp': datetime.now().isoformat()
    }

    for n_samples in samples_per_class:
        print(f'\n{"="*50}')
        print(f'Samples per class: {n_samples}')
        print(f'{"="*50}')

        for seed in seeds:
            print(f'\n--- Seed {seed} ---')

            # Get data loaders (64x64 images for relational task)
            train_loader, test_loader = get_relational_mnist_loaders(
                samples_per_class=n_samples,
                batch_size=min(64, n_samples * 2),
                image_size=64,
                seed=seed,
                n_distractors=n_distractors
            )
            print(f'Training samples: {len(train_loader.dataset)}')

            # Adjust epochs for very small datasets
            actual_epochs = epochs if n_samples >= 100 else min(epochs, 100)

            # Use smaller learning rate for small datasets
            lr = 0.001 if n_samples < 100 else 0.01

            # Train HC-Net (need to adjust for 64x64 images)
            print('Training HC-Net...')
            torch.manual_seed(seed)
            hcnet = HCNetResNetSmall(num_classes=2, block_size=8, base_width=32)
            hcnet_results = train_model(
                hcnet, train_loader, test_loader,
                epochs=actual_epochs, device=device, lr=lr
            )
            results['hcnet'][n_samples].append(hcnet_results)
            print(f'HC-Net Best: {hcnet_results["best_test_acc"]:.2f}%')

            # Train Baseline
            print('Training Baseline...')
            torch.manual_seed(seed)
            baseline = BaselineResNetSmall(num_classes=2, base_width=32)
            baseline_results = train_model(
                baseline, train_loader, test_loader,
                epochs=actual_epochs, device=device, lr=lr
            )
            results['baseline'][n_samples].append(baseline_results)
            print(f'Baseline Best: {baseline_results["best_test_acc"]:.2f}%')

    # Compute summary
    summary = compute_summary(results)
    results['summary'] = summary

    # Save results
    save_results(results, output_dir)

    # Plot results
    plot_sample_efficiency(summary, output_dir)
    plot_learning_curves(results, output_dir)

    # Print summary
    print_summary(summary)

    print(f'\nResults saved to {output_dir}')
    return results


def compute_summary(results: Dict) -> Dict:
    """Compute summary statistics."""
    summary = {
        'samples': results['samples'],
        'hcnet_mean': [],
        'hcnet_std': [],
        'baseline_mean': [],
        'baseline_std': [],
        'improvement': [],
        'hcnet_epochs_to_90': [],
        'baseline_epochs_to_90': []
    }

    for n in results['samples']:
        hcnet_accs = [r['best_test_acc'] for r in results['hcnet'][n]]
        baseline_accs = [r['best_test_acc'] for r in results['baseline'][n]]

        hcnet_e90 = [r['epochs_to_90'] for r in results['hcnet'][n] if r['epochs_to_90'] is not None]
        baseline_e90 = [r['epochs_to_90'] for r in results['baseline'][n] if r['epochs_to_90'] is not None]

        summary['hcnet_mean'].append(np.mean(hcnet_accs))
        summary['hcnet_std'].append(np.std(hcnet_accs))
        summary['baseline_mean'].append(np.mean(baseline_accs))
        summary['baseline_std'].append(np.std(baseline_accs))
        summary['improvement'].append(np.mean(hcnet_accs) - np.mean(baseline_accs))
        summary['hcnet_epochs_to_90'].append(np.mean(hcnet_e90) if hcnet_e90 else None)
        summary['baseline_epochs_to_90'].append(np.mean(baseline_e90) if baseline_e90 else None)

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


def plot_sample_efficiency(summary: Dict, output_dir: str):
    """Plot sample efficiency comparison."""
    samples = summary['samples']
    x_pos = np.arange(len(samples))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy comparison
    ax = axes[0]
    ax.errorbar(x_pos - 0.1, summary['hcnet_mean'], yerr=summary['hcnet_std'],
                label='HC-Net', marker='o', capsize=5, linewidth=2, markersize=8)
    ax.errorbar(x_pos + 0.1, summary['baseline_mean'], yerr=summary['baseline_std'],
                label='Baseline', marker='s', capsize=5, linewidth=2, markersize=8)

    ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% Target')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.3, label='Chance')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(n) for n in samples])
    ax.set_xlabel('Training Samples per Class')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Relational MNIST: 3-Way Binding Task')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([40, 105])

    # Improvement bar chart
    ax = axes[1]
    colors = ['green' if imp > 0 else 'red' for imp in summary['improvement']]
    bars = ax.bar(x_pos, summary['improvement'], color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(n) for n in samples])
    ax.set_xlabel('Training Samples per Class')
    ax.set_ylabel('HC-Net - Baseline Accuracy (%)')
    ax.set_title('HC-Net Advantage on Relational Task')
    ax.grid(True, alpha=0.3)

    for bar, val in zip(bars, summary['improvement']):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height > 0 else -10),
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_efficiency.png'), dpi=150)
    plt.close()


def plot_learning_curves(results: Dict, output_dir: str):
    """Plot learning curves."""
    samples_to_plot = [s for s in results['samples'] if s in [50, 500, 2000]]
    if not samples_to_plot:
        samples_to_plot = results['samples'][:3]

    fig, axes = plt.subplots(1, len(samples_to_plot), figsize=(5*len(samples_to_plot), 4))
    if len(samples_to_plot) == 1:
        axes = [axes]

    for ax, n_samples in zip(axes, samples_to_plot):
        hcnet_curves = [r['history']['test_acc'] for r in results['hcnet'][n_samples]]
        baseline_curves = [r['history']['test_acc'] for r in results['baseline'][n_samples]]

        epochs = range(len(hcnet_curves[0]))

        hcnet_mean = np.mean(hcnet_curves, axis=0)
        hcnet_std = np.std(hcnet_curves, axis=0)
        baseline_mean = np.mean(baseline_curves, axis=0)
        baseline_std = np.std(baseline_curves, axis=0)

        ax.plot(epochs, hcnet_mean, label='HC-Net', color='blue')
        ax.fill_between(epochs, hcnet_mean - hcnet_std, hcnet_mean + hcnet_std, alpha=0.2, color='blue')

        ax.plot(epochs, baseline_mean, label='Baseline', color='orange')
        ax.fill_between(epochs, baseline_mean - baseline_std, baseline_mean + baseline_std, alpha=0.2, color='orange')

        ax.axhline(y=90, color='green', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title(f'N={n_samples} samples/class')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Relational MNIST Learning Curves', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=150)
    plt.close()


def print_summary(summary: Dict):
    """Print summary table."""
    print('\n' + '='*80)
    print('SUMMARY: Relational MNIST (3-Way Binding)')
    print('Task: "Red Triangle LEFT-OF Blue Circle"')
    print('='*80)
    print(f'{"Samples":>10} {"HC-Net":>18} {"Baseline":>18} {"Improvement":>12}')
    print('-'*80)

    for i, n in enumerate(summary['samples']):
        hcnet_str = f'{summary["hcnet_mean"][i]:.1f} +/- {summary["hcnet_std"][i]:.1f}'
        base_str = f'{summary["baseline_mean"][i]:.1f} +/- {summary["baseline_std"][i]:.1f}'
        imp = summary["improvement"][i]
        imp_str = f'{imp:+.1f}%'
        marker = '***' if imp > 5 else ('**' if imp > 2 else ('*' if imp > 0 else ''))
        print(f'{n:>10} {hcnet_str:>18} {base_str:>18} {imp_str:>12} {marker}')

    print('='*80)

    # Find thresholds
    for i, (n, acc) in enumerate(zip(summary['samples'], summary['hcnet_mean'])):
        if acc >= 90:
            print(f'HC-Net reaches 90% at N={n} samples/class')
            break
    else:
        print('HC-Net did not reach 90% in tested range')

    for i, (n, acc) in enumerate(zip(summary['samples'], summary['baseline_mean'])):
        if acc >= 90:
            print(f'Baseline reaches 90% at N={n} samples/class')
            break
    else:
        print('Baseline did not reach 90% in tested range')

    print('='*80)
    print('*** = >5% advantage, ** = >2% advantage, * = marginal advantage')


def main():
    parser = argparse.ArgumentParser(description='Relational MNIST Experiment')
    parser.add_argument('--samples', type=int, nargs='+',
                        default=[10, 50, 100, 500, 1000, 2000],
                        help='Training samples per class')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[42, 123, 456],
                        help='Random seeds')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--output', type=str, default='./results/relational_mnist',
                        help='Output directory')
    parser.add_argument('--distractors', type=int, default=0,
                        help='Number of distractor objects (0=original task, 3=dense clutter)')
    args = parser.parse_args()

    run_relational_mnist_experiment(
        samples_per_class=args.samples,
        seeds=args.seeds,
        epochs=args.epochs,
        output_dir=args.output,
        n_distractors=args.distractors
    )


if __name__ == '__main__':
    main()
