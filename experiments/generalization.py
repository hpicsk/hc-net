"""
Compositional Generalization Experiment.

Tests whether PCNN better generalizes to unseen combinations
of attributes (superclasses) and objects (fine classes).

Usage:
    python -m experiments.generalization
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from data.compositional import CompositionallySplitCIFAR100
from models.pcnn_resnet import PCNNResNetSmall
from models.baseline_resnet import BaselineResNetSmall


def train_model_compositional(
    model: nn.Module,
    train_loader,
    test_seen_loader,
    test_holdout_loader,
    epochs: int = 100,
    device: str = 'cuda',
    lr: float = 0.1
) -> Dict:
    """Train model with compositional evaluation."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'train_acc': [], 'seen_acc': [], 'holdout_acc': [],
        'train_loss': [], 'seen_loss': [], 'holdout_loss': []
    }

    def evaluate(loader):
        model.eval()
        correct = 0
        total = 0
        total_loss = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return 100. * correct / total, total_loss / len(loader)

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

        # Evaluate on both test sets
        seen_acc, seen_loss = evaluate(test_seen_loader)
        holdout_acc, holdout_loss = evaluate(test_holdout_loader)

        scheduler.step()

        history['train_acc'].append(train_acc)
        history['seen_acc'].append(seen_acc)
        history['holdout_acc'].append(holdout_acc)
        history['train_loss'].append(train_loss)
        history['seen_loss'].append(seen_loss)
        history['holdout_loss'].append(holdout_loss)

        if epoch % 20 == 0:
            print(f'  Epoch {epoch}: Seen {seen_acc:.1f}%, Holdout {holdout_acc:.1f}%')

    return {
        'best_seen_acc': max(history['seen_acc']),
        'best_holdout_acc': max(history['holdout_acc']),
        'final_seen_acc': history['seen_acc'][-1],
        'final_holdout_acc': history['holdout_acc'][-1],
        'compositional_gap': max(history['seen_acc']) - max(history['holdout_acc']),
        'history': history
    }


def run_compositional_experiment(
    holdout_levels: List[int] = [1, 2],
    seeds: List[int] = [42, 123, 456],
    epochs: int = 100,
    output_dir: str = './results/compositional'
) -> Dict:
    """
    Run compositional generalization experiment.

    Args:
        holdout_levels: Number of fine classes to hold out per superclass
        seeds: Random seeds
        epochs: Training epochs
        output_dir: Output directory

    Returns:
        Results dictionary
    """
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = {
        'holdout_levels': holdout_levels,
        'seeds': seeds,
        'pcnn': {h: [] for h in holdout_levels},
        'baseline': {h: [] for h in holdout_levels}
    }

    for holdout in holdout_levels:
        print(f'\n=== Holdout Level: {holdout} per superclass ===')

        for seed in seeds:
            print(f'\n--- Seed {seed} ---')

            # Create compositional split
            data = CompositionallySplitCIFAR100(
                holdout_per_superclass=holdout,
                batch_size=128,
                seed=seed
            )

            train_loader = data.get_train_loader()
            test_seen_loader = data.get_test_seen_loader()
            test_holdout_loader = data.get_test_holdout_loader()

            stats = data.get_split_statistics()
            print(f'Split: {stats["num_seen_classes"]} seen, '
                  f'{stats["num_holdout_classes"]} holdout classes')
            print(f'Holdout classes: {stats["holdout_class_names"][:5]}...')

            # Train PCNN
            print('Training PCNN...')
            torch.manual_seed(seed)
            pcnn = PCNNResNetSmall(num_classes=100, block_size=8)
            pcnn_results = train_model_compositional(
                pcnn, train_loader, test_seen_loader, test_holdout_loader,
                epochs=epochs, device=device
            )
            pcnn_results['holdout_classes'] = stats['holdout_class_names']
            results['pcnn'][holdout].append(pcnn_results)
            print(f'PCNN Seen: {pcnn_results["best_seen_acc"]:.2f}%, '
                  f'Holdout: {pcnn_results["best_holdout_acc"]:.2f}%')

            # Train Baseline
            print('Training Baseline...')
            torch.manual_seed(seed)
            baseline = BaselineResNetSmall(num_classes=100)
            baseline_results = train_model_compositional(
                baseline, train_loader, test_seen_loader, test_holdout_loader,
                epochs=epochs, device=device
            )
            baseline_results['holdout_classes'] = stats['holdout_class_names']
            results['baseline'][holdout].append(baseline_results)
            print(f'Baseline Seen: {baseline_results["best_seen_acc"]:.2f}%, '
                  f'Holdout: {baseline_results["best_holdout_acc"]:.2f}%')

    # Compute summary
    summary = compute_summary(results)
    results['summary'] = summary

    # Save results
    results_path = os.path.join(output_dir, 'results.json')

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
    """Compute summary statistics."""
    summary = {
        'holdout_levels': results['holdout_levels'],
        'pcnn_seen_mean': [],
        'pcnn_seen_std': [],
        'pcnn_holdout_mean': [],
        'pcnn_holdout_std': [],
        'baseline_seen_mean': [],
        'baseline_seen_std': [],
        'baseline_holdout_mean': [],
        'baseline_holdout_std': [],
        'holdout_improvement': [],
        'gap_reduction': []
    }

    for h in results['holdout_levels']:
        # PCNN
        pcnn_seen = [r['best_seen_acc'] for r in results['pcnn'][h]]
        pcnn_holdout = [r['best_holdout_acc'] for r in results['pcnn'][h]]
        pcnn_gap = [r['compositional_gap'] for r in results['pcnn'][h]]

        summary['pcnn_seen_mean'].append(np.mean(pcnn_seen))
        summary['pcnn_seen_std'].append(np.std(pcnn_seen))
        summary['pcnn_holdout_mean'].append(np.mean(pcnn_holdout))
        summary['pcnn_holdout_std'].append(np.std(pcnn_holdout))

        # Baseline
        baseline_seen = [r['best_seen_acc'] for r in results['baseline'][h]]
        baseline_holdout = [r['best_holdout_acc'] for r in results['baseline'][h]]
        baseline_gap = [r['compositional_gap'] for r in results['baseline'][h]]

        summary['baseline_seen_mean'].append(np.mean(baseline_seen))
        summary['baseline_seen_std'].append(np.std(baseline_seen))
        summary['baseline_holdout_mean'].append(np.mean(baseline_holdout))
        summary['baseline_holdout_std'].append(np.std(baseline_holdout))

        # Improvements
        summary['holdout_improvement'].append(
            np.mean(pcnn_holdout) - np.mean(baseline_holdout)
        )
        summary['gap_reduction'].append(
            np.mean(baseline_gap) - np.mean(pcnn_gap)
        )

    return summary


def plot_results(summary: Dict, output_dir: str):
    """Plot compositional generalization results."""
    holdout_levels = summary['holdout_levels']
    x = np.arange(len(holdout_levels))
    width = 0.35

    # Comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Seen vs Holdout accuracy
    ax = axes[0]
    ax.bar(x - width/2, summary['pcnn_holdout_mean'], width, label='PCNN Holdout',
           yerr=summary['pcnn_holdout_std'], capsize=5, color='#2ecc71')
    ax.bar(x + width/2, summary['baseline_holdout_mean'], width, label='Baseline Holdout',
           yerr=summary['baseline_holdout_std'], capsize=5, color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h} per class' for h in holdout_levels])
    ax.set_xlabel('Holdout Level')
    ax.set_ylabel('Holdout Accuracy (%)')
    ax.set_title('Zero-Shot Compositional Generalization')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Improvement
    ax = axes[1]
    colors = ['green' if imp > 0 else 'red' for imp in summary['holdout_improvement']]
    ax.bar(x, summary['holdout_improvement'], color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h} per class' for h in holdout_levels])
    ax.set_xlabel('Holdout Level')
    ax.set_ylabel('PCNN Improvement (%)')
    ax.set_title('PCNN vs Baseline on Holdout Classes')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'compositional.png'), dpi=150)
    plt.close()

    # Compositional gap comparison
    fig, ax = plt.subplots(figsize=(8, 5))

    width = 0.35
    pcnn_gaps = [s - h for s, h in zip(summary['pcnn_seen_mean'],
                                        summary['pcnn_holdout_mean'])]
    baseline_gaps = [s - h for s, h in zip(summary['baseline_seen_mean'],
                                            summary['baseline_holdout_mean'])]

    ax.bar(x - width/2, pcnn_gaps, width, label='PCNN', color='#3498db')
    ax.bar(x + width/2, baseline_gaps, width, label='Baseline', color='#e67e22')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h} per class' for h in holdout_levels])
    ax.set_xlabel('Holdout Level')
    ax.set_ylabel('Compositional Gap (Seen - Holdout) %')
    ax.set_title('Compositional Gap Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'compositional_gap.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--holdout', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--output', type=str, default='./results/compositional')
    args = parser.parse_args()

    run_compositional_experiment(
        holdout_levels=args.holdout,
        seeds=args.seeds,
        epochs=args.epochs,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
