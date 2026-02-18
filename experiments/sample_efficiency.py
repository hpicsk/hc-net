"""
Sample Efficiency Experiment.

Compares PCNN and baseline models on varying amounts of training data.
Tests the hypothesis that PCNN's inductive bias leads to better
generalization with limited data.

Usage:
    python -m experiments.sample_efficiency
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

from pcnn.data.cifar100 import get_cifar100_loaders
from pcnn.models.pcnn_resnet import PCNNResNetSmall
from pcnn.models.baseline_resnet import BaselineResNetSmall


def train_model(
    model: nn.Module,
    train_loader,
    test_loader,
    epochs: int = 100,
    device: str = 'cuda',
    lr: float = 0.1
) -> Dict:
    """Train model and return results."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}

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

        scheduler.step()

        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)

        if epoch % 20 == 0:
            print(f'  Epoch {epoch}: Train {train_acc:.1f}%, Test {test_acc:.1f}%')

    return {
        'best_test_acc': max(history['test_acc']),
        'final_test_acc': history['test_acc'][-1],
        'best_train_acc': max(history['train_acc']),
        'generalization_gap': max(history['train_acc']) - max(history['test_acc']),
        'history': history
    }


def run_sample_efficiency_experiment(
    fractions: List[float] = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
    seeds: List[int] = [42, 123, 456],
    epochs: int = 100,
    output_dir: str = './results/sample_efficiency'
) -> Dict:
    """
    Run sample efficiency experiment.

    Args:
        fractions: Training data fractions to test
        seeds: Random seeds for multiple runs
        epochs: Training epochs per run
        output_dir: Directory to save results

    Returns:
        Dictionary with all results
    """
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = {
        'fractions': fractions,
        'seeds': seeds,
        'pcnn': {f: [] for f in fractions},
        'baseline': {f: [] for f in fractions}
    }

    for fraction in fractions:
        print(f'\n=== Data Fraction: {fraction:.0%} ===')

        for seed in seeds:
            print(f'\n--- Seed {seed} ---')

            # Get data loaders
            train_loader, test_loader = get_cifar100_loaders(
                batch_size=128,
                subset_fraction=fraction,
                seed=seed
            )
            print(f'Training samples: {len(train_loader.dataset)}')

            # Train PCNN
            print('Training PCNN...')
            torch.manual_seed(seed)
            pcnn = PCNNResNetSmall(num_classes=100, block_size=8)
            pcnn_results = train_model(pcnn, train_loader, test_loader,
                                       epochs=epochs, device=device)
            results['pcnn'][fraction].append(pcnn_results)
            print(f'PCNN Best: {pcnn_results["best_test_acc"]:.2f}%')

            # Train Baseline
            print('Training Baseline...')
            torch.manual_seed(seed)
            baseline = BaselineResNetSmall(num_classes=100)
            baseline_results = train_model(baseline, train_loader, test_loader,
                                          epochs=epochs, device=device)
            results['baseline'][fraction].append(baseline_results)
            print(f'Baseline Best: {baseline_results["best_test_acc"]:.2f}%')

    # Compute summary statistics
    summary = compute_summary(results)
    results['summary'] = summary

    # Save results
    results_path = os.path.join(output_dir, 'results.json')

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    with open(results_path, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)

    # Plot results
    plot_results(summary, output_dir)

    print(f'\nResults saved to {output_dir}')
    return results


def compute_summary(results: Dict) -> Dict:
    """Compute summary statistics from results."""
    summary = {
        'fractions': results['fractions'],
        'pcnn_mean': [],
        'pcnn_std': [],
        'baseline_mean': [],
        'baseline_std': [],
        'improvement': [],
        'pcnn_gap_mean': [],
        'baseline_gap_mean': []
    }

    for f in results['fractions']:
        pcnn_accs = [r['best_test_acc'] for r in results['pcnn'][f]]
        baseline_accs = [r['best_test_acc'] for r in results['baseline'][f]]

        pcnn_gaps = [r['generalization_gap'] for r in results['pcnn'][f]]
        baseline_gaps = [r['generalization_gap'] for r in results['baseline'][f]]

        summary['pcnn_mean'].append(np.mean(pcnn_accs))
        summary['pcnn_std'].append(np.std(pcnn_accs))
        summary['baseline_mean'].append(np.mean(baseline_accs))
        summary['baseline_std'].append(np.std(baseline_accs))
        summary['improvement'].append(np.mean(pcnn_accs) - np.mean(baseline_accs))
        summary['pcnn_gap_mean'].append(np.mean(pcnn_gaps))
        summary['baseline_gap_mean'].append(np.mean(baseline_gaps))

    return summary


def plot_results(summary: Dict, output_dir: str):
    """Plot sample efficiency results."""
    fractions = summary['fractions']
    x_pos = np.arange(len(fractions))

    # Accuracy comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Test accuracy
    ax = axes[0]
    ax.errorbar(x_pos - 0.1, summary['pcnn_mean'], yerr=summary['pcnn_std'],
                label='PCNN', marker='o', capsize=5)
    ax.errorbar(x_pos + 0.1, summary['baseline_mean'], yerr=summary['baseline_std'],
                label='Baseline', marker='s', capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{f:.0%}' for f in fractions])
    ax.set_xlabel('Training Data Fraction')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Sample Efficiency Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Improvement
    ax = axes[1]
    colors = ['green' if imp > 0 else 'red' for imp in summary['improvement']]
    ax.bar(x_pos, summary['improvement'], color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{f:.0%}' for f in fractions])
    ax.set_xlabel('Training Data Fraction')
    ax.set_ylabel('PCNN Improvement (%)')
    ax.set_title('PCNN vs Baseline Improvement')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_efficiency.png'), dpi=150)
    plt.close()

    # Generalization gap comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fractions, summary['pcnn_gap_mean'], 'o-', label='PCNN')
    ax.plot(fractions, summary['baseline_gap_mean'], 's-', label='Baseline')
    ax.set_xlabel('Training Data Fraction')
    ax.set_ylabel('Generalization Gap (%)')
    ax.set_title('Generalization Gap (Train - Test Accuracy)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'generalization_gap.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fractions', type=float, nargs='+',
                        default=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[42, 123, 456])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--output', type=str, default='./results/sample_efficiency')
    args = parser.parse_args()

    run_sample_efficiency_experiment(
        fractions=args.fractions,
        seeds=args.seeds,
        epochs=args.epochs,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
