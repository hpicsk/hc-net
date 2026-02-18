"""
Generate publication-quality figures for HC-Net paper.
Creates figures from scaling analysis and ablation study results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import argparse


def load_json(path: str) -> dict:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def create_scaling_figure(scaling_data: dict, output_dir: Path):
    """
    Create scaling analysis figure showing:
    - Left: Forward time vs N (log-log)
    - Right: Memory usage vs N (log-log)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    results = scaling_data['results']
    n_particles = scaling_data['config']['n_particles_list']

    # Color scheme
    colors = {
        'hcnet': '#2E86AB',      # Blue
        'egnn': '#E94F37',       # Red
        'cgenn': '#F4A261',      # Orange
        'nequip': '#8338EC',     # Purple
        'baseline': '#95A5A6'    # Gray
    }

    labels = {
        'hcnet': 'HC-Net',
        'egnn': 'EGNN',
        'cgenn': 'CGENN',
        'nequip': 'NequIP',
        'baseline': 'MLP Baseline'
    }

    markers = {'hcnet': 'o', 'egnn': 's', 'cgenn': '^', 'nequip': 'D', 'baseline': 'x'}

    # Left plot: Forward time
    ax1 = axes[0]
    for model_name, model_data in results.items():
        times = model_data['forward_times']
        n_list = sorted([int(k) for k in times.keys()])
        time_list = [times[str(n)] for n in n_list]

        alpha = model_data.get('complexity', {}).get('alpha', 0)
        label = f"{labels[model_name]} (O(N^{{{alpha:.2f}}}))"

        ax1.loglog(n_list, time_list,
                   marker=markers[model_name],
                   color=colors[model_name],
                   linewidth=2, markersize=8,
                   label=label)

    ax1.set_xlabel('Number of Particles (N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Forward Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Computational Complexity', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3, which='both')

    # Right plot: Memory usage
    ax2 = axes[1]
    for model_name, model_data in results.items():
        memory = model_data['peak_memory_mb']
        n_list = sorted([int(k) for k in memory.keys()])
        mem_list = [memory[str(n)] for n in n_list]

        ax2.loglog(n_list, mem_list,
                   marker=markers[model_name],
                   color=colors[model_name],
                   linewidth=2, markersize=8,
                   label=labels[model_name])

    ax2.set_xlabel('Number of Particles (N)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Peak Memory (MB)', fontsize=12, fontweight='bold')
    ax2.set_title('Memory Efficiency', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    # Save
    output_path = output_dir / 'figure_scaling.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    output_path_png = output_dir / 'figure_scaling.png'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Saved: {output_path}')
    print(f'Saved: {output_path_png}')

    # Print summary table
    print('\n--- Scaling Summary (N=100) ---')
    print(f'{"Model":<12} {"Time (ms)":<12} {"Memory (MB)":<15} {"Complexity":<12}')
    print('-' * 55)
    for model_name, model_data in results.items():
        time_100 = model_data['forward_times'].get('100', 'N/A')
        mem_100 = model_data['peak_memory_mb'].get('100', 'N/A')
        alpha = model_data.get('complexity', {}).get('alpha', 0)
        print(f'{labels[model_name]:<12} {time_100:<12.2f} {mem_100:<15.2f} O(N^{alpha:.2f})')


def create_ablation_figure(ablation_data: dict, output_dir: Path):
    """
    Create ablation study figure showing:
    - Left: Component ablation bar chart
    - Right: Layer/dimension sensitivity
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    summary = ablation_data['summary']
    results = ablation_data['results']

    # Left: Component ablation
    ax1 = axes[0]

    variants = ['full', 'no_residual', 'no_geo_interaction', 'no_geo_mixing', 'no_layer_norm', 'with_attention']
    variant_labels = {
        'full': 'Full Model',
        'no_residual': 'w/o Residual',
        'no_geo_interaction': 'w/o Geo-Interact',
        'no_geo_mixing': 'w/o Geo-Mixing',
        'no_layer_norm': 'w/o LayerNorm',
        'with_attention': 'w/ Attention'
    }

    mse_values = []
    mse_stds = []
    labels = []
    colors = []

    base_mse = summary['full']['test_mse_mean']

    for v in variants:
        if v in summary:
            mse_values.append(summary[v]['test_mse_mean'] * 1e5)  # Scale to 1e-5
            mse_stds.append(summary[v]['test_mse_std'] * 1e5)
            labels.append(variant_labels[v])

            # Color based on impact
            delta = (summary[v]['test_mse_mean'] - base_mse) / base_mse * 100
            if v == 'full':
                colors.append('#2E86AB')  # Blue for baseline
            elif delta > 5:
                colors.append('#E94F37')  # Red for significant degradation
            elif delta < -1:
                colors.append('#2ECC71')  # Green for improvement
            else:
                colors.append('#95A5A6')  # Gray for minimal impact

    x = np.arange(len(labels))
    bars = ax1.bar(x, mse_values, yerr=mse_stds, capsize=4,
                   color=colors, edgecolor='black', linewidth=1)

    ax1.set_ylabel('Test MSE (×10⁻⁵)', fontsize=12, fontweight='bold')
    ax1.set_title('Component Ablation', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax1.axhline(y=base_mse * 1e5, color='#2E86AB', linestyle='--', alpha=0.7, label='Full Model')
    ax1.grid(axis='y', alpha=0.3)

    # Add percentage annotations
    for i, (v, bar) in enumerate(zip(variants, bars)):
        if v != 'full' and v in summary:
            delta = (summary[v]['test_mse_mean'] - base_mse) / base_mse * 100
            sign = '+' if delta > 0 else ''
            ax1.annotate(f'{sign}{delta:.1f}%',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=9, fontweight='bold')

    # Right: Layer count sensitivity
    ax2 = axes[1]

    layer_results = [r for r in results if r['model'].startswith('layers_')]
    if layer_results:
        n_layers = [r['n_layers'] for r in layer_results]
        layer_mse = [r['best_test_loss'] * 1e5 for r in layer_results]

        ax2.plot(n_layers, layer_mse, 'o-', color='#2E86AB',
                linewidth=2, markersize=10, label='Test MSE')
        ax2.set_xlabel('Number of Layers', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Test MSE (×10⁻⁵)', fontsize=12, fontweight='bold')
        ax2.set_title('Layer Sensitivity', fontsize=14, fontweight='bold')
        ax2.set_xticks(n_layers)
        ax2.grid(True, alpha=0.3)

        # Add secondary axis for hidden dim
        hidden_results = [r for r in results if r['model'].startswith('hidden_')]
        if hidden_results:
            ax2_twin = ax2.twinx()
            hidden_dims = [r['hidden_dim'] for r in hidden_results]
            hidden_mse = [r['best_test_loss'] * 1e5 for r in hidden_results]
            ax2_twin.plot(range(1, len(hidden_dims)+1), hidden_mse, 's--',
                         color='#E94F37', linewidth=2, markersize=8, label='Hidden Dim')
            ax2_twin.set_ylabel('Hidden Dim MSE (×10⁻⁵)', fontsize=10, color='#E94F37')
            ax2_twin.tick_params(axis='y', labelcolor='#E94F37')

    plt.tight_layout()

    # Save
    output_path = output_dir / 'figure_ablation.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    output_path_png = output_dir / 'figure_ablation.png'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Saved: {output_path}')
    print(f'Saved: {output_path_png}')

    # Print ablation summary
    print('\n--- Ablation Study Summary ---')
    print(f'{"Variant":<20} {"MSE":<15} {"Δ%":<10}')
    print('-' * 45)
    for v in variants:
        if v in summary:
            mse = summary[v]['test_mse_mean']
            delta = (mse - base_mse) / base_mse * 100
            sign = '+' if delta > 0 else ''
            print(f'{variant_labels[v]:<20} {mse:.6f}   {sign}{delta:.1f}%')


def create_combined_summary_figure(scaling_data: dict, ablation_data: dict, output_dir: Path):
    """
    Create a 2x2 summary figure for the paper.
    """
    fig = plt.figure(figsize=(14, 10))

    results = scaling_data['results']
    summary = ablation_data['summary']

    colors = {
        'hcnet': '#2E86AB', 'egnn': '#E94F37', 'cgenn': '#F4A261',
        'nequip': '#8338EC', 'baseline': '#95A5A6'
    }
    labels = {
        'hcnet': 'HC-Net', 'egnn': 'EGNN', 'cgenn': 'CGENN',
        'nequip': 'NequIP', 'baseline': 'MLP'
    }
    markers = {'hcnet': 'o', 'egnn': 's', 'cgenn': '^', 'nequip': 'D', 'baseline': 'x'}

    # (a) Computational scaling
    ax1 = fig.add_subplot(2, 2, 1)
    for model_name, model_data in results.items():
        times = model_data['forward_times']
        n_list = sorted([int(k) for k in times.keys()])
        time_list = [times[str(n)] for n in n_list]
        ax1.loglog(n_list, time_list, marker=markers[model_name],
                   color=colors[model_name], linewidth=2, markersize=7,
                   label=labels[model_name])
    ax1.set_xlabel('N', fontsize=11)
    ax1.set_ylabel('Time (ms)', fontsize=11)
    ax1.set_title('(a) Computational Scaling', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3, which='both')

    # (b) Memory efficiency
    ax2 = fig.add_subplot(2, 2, 2)
    for model_name, model_data in results.items():
        memory = model_data['peak_memory_mb']
        n_list = sorted([int(k) for k in memory.keys()])
        mem_list = [memory[str(n)] for n in n_list]
        ax2.loglog(n_list, mem_list, marker=markers[model_name],
                   color=colors[model_name], linewidth=2, markersize=7,
                   label=labels[model_name])
    ax2.set_xlabel('N', fontsize=11)
    ax2.set_ylabel('Memory (MB)', fontsize=11)
    ax2.set_title('(b) Memory Usage', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(True, alpha=0.3, which='both')

    # (c) Component ablation
    ax3 = fig.add_subplot(2, 2, 3)
    variants = ['full', 'no_residual', 'no_geo_interaction', 'no_geo_mixing', 'no_layer_norm', 'with_attention']
    variant_labels_short = ['Full', 'w/o Res', 'w/o Geo-I', 'w/o Geo-M', 'w/o LN', 'w/ Attn']

    base_mse = summary['full']['test_mse_mean']
    mse_values = [(summary[v]['test_mse_mean'] - base_mse) / base_mse * 100
                  for v in variants if v in summary]

    bar_colors = ['#2E86AB' if v == 'full' else
                  ('#E94F37' if m > 5 else '#95A5A6')
                  for v, m in zip(variants, mse_values)]

    x = np.arange(len(variant_labels_short))
    ax3.bar(x, mse_values, color=bar_colors, edgecolor='black', linewidth=1)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('Δ MSE (%)', fontsize=11)
    ax3.set_title('(c) Component Ablation', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(variant_labels_short, rotation=45, ha='right', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    # (d) Speedup comparison at N=100
    ax4 = fig.add_subplot(2, 2, 4)
    hcnet_time = results['hcnet']['forward_times']['100']
    speedups = []
    model_names = []
    for m in ['egnn', 'cgenn', 'nequip']:
        if m in results:
            speedup = results[m]['forward_times']['100'] / hcnet_time
            speedups.append(speedup)
            model_names.append(labels[m])

    bar_colors = ['#E94F37', '#F4A261', '#8338EC'][:len(speedups)]
    ax4.bar(model_names, speedups, color=bar_colors, edgecolor='black', linewidth=1)
    ax4.axhline(y=1, color='#2E86AB', linestyle='--', linewidth=2, label='HC-Net baseline')
    ax4.set_ylabel('Speedup vs HC-Net', fontsize=11)
    ax4.set_title('(d) Relative Speed (N=100)', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    # Add speedup annotations
    for i, (name, speedup) in enumerate(zip(model_names, speedups)):
        ax4.annotate(f'{speedup:.1f}x slower', xy=(i, speedup),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = output_dir / 'figure_summary.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    output_path_png = output_dir / 'figure_summary.png'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Saved: {output_path}')
    print(f'Saved: {output_path_png}')


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--scaling', type=str,
                        default='./results/scaling/scaling_20260207_115019.json',
                        help='Path to scaling results')
    parser.add_argument('--ablation', type=str,
                        default='./results/ablation/ablation_20260207_121205.json',
                        help='Path to ablation results')
    parser.add_argument('--output', type=str, default='./figures',
                        help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Loading results...')
    scaling_data = load_json(args.scaling)
    ablation_data = load_json(args.ablation)

    print('\nGenerating figures...')

    # Individual figures
    create_scaling_figure(scaling_data, output_dir)
    create_ablation_figure(ablation_data, output_dir)

    # Combined summary figure
    create_combined_summary_figure(scaling_data, ablation_data, output_dir)

    print('\nDone!')


if __name__ == '__main__':
    main()
