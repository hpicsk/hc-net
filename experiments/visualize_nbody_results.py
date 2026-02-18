"""
Visualization script for Starved N-Body experiment results.
Creates publication-quality figures for the HC-Net paper.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path
import argparse


def load_results(results_path: str) -> dict:
    """Load experiment results from JSON."""
    with open(results_path, 'r') as f:
        return json.load(f)


def create_ood_bar_chart(results: dict, output_dir: Path, n_train: int = 100, n_particles: int = 3):
    """
    Create the "killer" bar chart showing OOD generalization gap.

    Shows MSE at 0°, 45°, 90° rotation angles for Clifford vs Baseline.
    """
    angles = results['rotation_angles']
    n_seeds = len(results['seeds'])

    # Extract data for specified configuration
    clifford_data = results['clifford'][str(n_train)][str(n_particles)]
    baseline_data = results['baseline'][str(n_train)][str(n_particles)]

    # Compute mean and std for each angle
    clifford_means = []
    clifford_stds = []
    baseline_means = []
    baseline_stds = []

    for angle in angles:
        angle_key = str(float(angle))

        cliff_losses = [run['final_ood_losses'][angle_key] for run in clifford_data]
        base_losses = [run['final_ood_losses'][angle_key] for run in baseline_data]

        clifford_means.append(np.mean(cliff_losses))
        clifford_stds.append(np.std(cliff_losses))
        baseline_means.append(np.mean(base_losses))
        baseline_stds.append(np.std(base_losses))

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(angles))
    width = 0.35

    # Plot bars
    bars1 = ax.bar(x - width/2, clifford_means, width,
                   yerr=clifford_stds, label='HC-Net (Clifford)',
                   color='#2E86AB', capsize=5, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, baseline_means, width,
                   yerr=baseline_stds, label='Baseline MLP',
                   color='#E94F37', capsize=5, edgecolor='black', linewidth=1)

    # Formatting
    ax.set_xlabel('Rotation Angle (Test)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
    ax.set_title(f'Out-of-Distribution Generalization\n(N={n_train} training samples, {n_particles} particles)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(a)}°' for a in angles], fontsize=11)
    ax.legend(fontsize=11, loc='upper left')

    # Add improvement annotations
    for i, (cliff_m, base_m) in enumerate(zip(clifford_means, baseline_means)):
        if base_m > cliff_m and base_m > 0:
            improvement = (base_m - cliff_m) / base_m * 100
            ratio = cliff_m / base_m if base_m > 0 else 1.0
            if improvement > 10:  # Only annotate significant improvements
                y_pos = max(cliff_m, base_m) + max(clifford_stds[i], baseline_stds[i]) * 1.5
                ax.annotate(f'{improvement:.0f}% better\n({ratio:.2f}x)',
                           xy=(x[i], y_pos), ha='center', fontsize=9,
                           fontweight='bold', color='#2E86AB')

    # Use log scale if there's a large range
    max_val = max(max(baseline_means), max(clifford_means))
    min_val = min(min(baseline_means), min(clifford_means))
    if max_val / (min_val + 1e-10) > 10:
        ax.set_yscale('log')
        ax.set_ylabel('Mean Squared Error (log scale)', fontsize=12, fontweight='bold')

    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save in multiple formats
    output_path_pdf = output_dir / 'figure1_ood_bar.pdf'
    output_path_png = output_dir / 'figure1_ood_bar.png'

    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Saved: {output_path_pdf}')
    print(f'Saved: {output_path_png}')

    # Print statistics
    print(f'\n--- Statistics for N={n_train}, particles={n_particles} ---')
    for i, angle in enumerate(angles):
        ratio = clifford_means[i] / baseline_means[i] if baseline_means[i] > 0 else 1.0
        print(f'  {int(angle)}°: Clifford={clifford_means[i]:.6f} ± {clifford_stds[i]:.6f}, '
              f'Baseline={baseline_means[i]:.6f} ± {baseline_stds[i]:.6f}, Ratio={ratio:.3f}')


def create_combined_ood_chart(results: dict, output_dir: Path, n_train: int = 100):
    """
    Create a multi-panel chart showing OOD performance across particle counts.
    """
    angles = results['rotation_angles']
    n_particles_list = results['n_particles']

    fig, axes = plt.subplots(1, len(n_particles_list), figsize=(4*len(n_particles_list), 4), sharey=True)
    if len(n_particles_list) == 1:
        axes = [axes]

    for ax, n_particles in zip(axes, n_particles_list):
        clifford_data = results['clifford'][str(n_train)][str(n_particles)]
        baseline_data = results['baseline'][str(n_train)][str(n_particles)]

        clifford_means = []
        clifford_stds = []
        baseline_means = []
        baseline_stds = []

        for angle in angles:
            angle_key = str(float(angle))
            cliff_losses = [run['final_ood_losses'][angle_key] for run in clifford_data]
            base_losses = [run['final_ood_losses'][angle_key] for run in baseline_data]

            clifford_means.append(np.mean(cliff_losses))
            clifford_stds.append(np.std(cliff_losses))
            baseline_means.append(np.mean(base_losses))
            baseline_stds.append(np.std(base_losses))

        x = np.arange(len(angles))
        width = 0.35

        ax.bar(x - width/2, clifford_means, width, yerr=clifford_stds,
               label='HC-Net', color='#2E86AB', capsize=3, edgecolor='black', linewidth=0.5)
        ax.bar(x + width/2, baseline_means, width, yerr=baseline_stds,
               label='Baseline', color='#E94F37', capsize=3, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Rotation Angle', fontsize=10)
        ax.set_title(f'{n_particles} Particles', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(a)}°' for a in angles], fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('MSE', fontsize=10, fontweight='bold')
    axes[-1].legend(fontsize=9, loc='upper left')

    fig.suptitle(f'OOD Generalization with N={n_train} Training Samples', fontsize=12, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'figure1_ood_combined.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def create_summary_table(results: dict, output_dir: Path):
    """Create a LaTeX-formatted summary table."""

    table_lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{OOD Generalization Performance (MSE)}',
        r'\begin{tabular}{llrrr}',
        r'\toprule',
        r'N & Particles & 0° & 45° & 90° \\',
        r'\midrule',
    ]

    for n_train in results['n_train_list']:
        for n_particles in results['n_particles']:
            clifford_data = results['clifford'][str(n_train)][str(n_particles)]
            baseline_data = results['baseline'][str(n_train)][str(n_particles)]

            # Clifford row
            cliff_vals = []
            for angle in results['rotation_angles']:
                losses = [run['final_ood_losses'][str(float(angle))] for run in clifford_data]
                cliff_vals.append(f'{np.mean(losses):.4f}')
            table_lines.append(f'{n_train} & {n_particles} (HC-Net) & {" & ".join(cliff_vals)} \\\\')

            # Baseline row
            base_vals = []
            for angle in results['rotation_angles']:
                losses = [run['final_ood_losses'][str(float(angle))] for run in baseline_data]
                base_vals.append(f'{np.mean(losses):.4f}')
            table_lines.append(f' & {n_particles} (Baseline) & {" & ".join(base_vals)} \\\\')

        table_lines.append(r'\midrule')

    table_lines[-1] = r'\bottomrule'
    table_lines.extend([
        r'\end{tabular}',
        r'\label{tab:ood}',
        r'\end{table}',
    ])

    output_path = output_dir / 'table_ood.tex'
    with open(output_path, 'w') as f:
        f.write('\n'.join(table_lines))
    print(f'Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Visualize Starved N-Body results')
    parser.add_argument('--results', type=str, default='./results/starved_nbody/results.json',
                        help='Path to results JSON')
    parser.add_argument('--output', type=str, default='./results/starved_nbody',
                        help='Output directory')
    parser.add_argument('--n-train', type=int, default=100,
                        help='Training samples to visualize')
    parser.add_argument('--n-particles', type=int, default=3,
                        help='Number of particles to visualize')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading results from {args.results}...')
    results = load_results(args.results)

    print(f'\nCreating visualizations...')

    # Main bar chart
    create_ood_bar_chart(results, output_dir, n_train=args.n_train, n_particles=args.n_particles)

    # Combined chart across particle counts
    create_combined_ood_chart(results, output_dir, n_train=args.n_train)

    # LaTeX table
    create_summary_table(results, output_dir)

    print('\nDone!')


if __name__ == '__main__':
    main()
