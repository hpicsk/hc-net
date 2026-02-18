"""
Generate publication-quality figures for Chiral-Global HC-Net paper.

Figures:
1. fig_grade_hierarchy.pdf  - Grade hierarchy accuracy (spiral + rotation)
2. fig_md17_molecules.pdf   - MD17 test MSE per molecule per model
3. fig_scaling.pdf          - Log-log forward time vs N
4. fig_ood_rotation.pdf     - OOD MSE vs rotation angle per model
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import glob as globmod


# --- Style ---
COLORS = {
    'hybrid_hcnet': '#2E86AB',
    'clifford_net': '#E94F37',
    'egnn': '#F4A261',
    'baseline': '#95A5A6',
    'clifford3d': '#E94F37',
    'baseline3d': '#95A5A6',
}

LABELS = {
    'hybrid_hcnet': 'Hybrid HC-Net (Ours)',
    'clifford_net': 'CliffordNet',
    'egnn': 'EGNN',
    'baseline': 'MLP Baseline',
    'clifford3d': 'CliffordNet 3D',
    'baseline3d': 'MLP Baseline',
}

MARKERS = {
    'hybrid_hcnet': 'o',
    'clifford_net': 's',
    'egnn': '^',
    'baseline': 'x',
    'clifford3d': 's',
    'baseline3d': 'x',
}


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def find_latest(pattern, results_dir):
    """Find the latest result file matching a pattern."""
    files = sorted(globmod.glob(str(results_dir / pattern)))
    if not files:
        return None
    return files[-1]


# ==========================================================================
# Figure 1: Grade Hierarchy
# ==========================================================================
def create_grade_hierarchy_figure(spiral_data, rotation_data, output_dir):
    """
    Grouped bar chart: accuracy per grade for chirality (spiral) and
    rotation tasks.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    grades = ['vector_only', 'bivector', 'trivector', 'full_clifford', 'learned']
    grade_labels = ['Vector\n(Grade 0+1)', 'Bivector\n(Grade 0-2)',
                    'Trivector\n(Grade 3)', 'Full Cl(3,0)\n(All Grades)',
                    'Learned\nProjection']

    spiral_means = [spiral_data['summary'][g]['mean'] for g in grades]
    spiral_stds = [spiral_data['summary'][g]['std'] for g in grades]
    rotation_means = [rotation_data['summary'][g]['mean'] for g in grades]
    rotation_stds = [rotation_data['summary'][g]['std'] for g in grades]

    x = np.arange(len(grades))
    width = 0.35

    bars1 = ax.bar(x - width/2, spiral_means, width, yerr=spiral_stds,
                   label='Chirality (Spiral)', color='#2E86AB',
                   edgecolor='black', linewidth=0.8, capsize=4)
    bars2 = ax.bar(x + width/2, rotation_means, width, yerr=rotation_stds,
                   label='Rotation Direction', color='#E94F37',
                   edgecolor='black', linewidth=0.8, capsize=4)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Chance')
    ax.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('Grade Hierarchy in Cl(3,0): Which Grades Encode What?',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(grade_labels, fontsize=10)
    ax.set_ylim(0.0, 1.15)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    # Annotate key findings
    for i, (s, r) in enumerate(zip(spiral_means, rotation_means)):
        if s > 0.95:
            ax.annotate(f'{s:.0%}', xy=(x[i] - width/2, s),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=9, fontweight='bold', color='#2E86AB')
        if r > 0.95:
            ax.annotate(f'{r:.0%}', xy=(x[i] + width/2, r),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=9, fontweight='bold', color='#E94F37')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(output_dir / f'fig_grade_hierarchy.{ext}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: fig_grade_hierarchy.pdf')


# ==========================================================================
# Figure 2: MD17 Molecules
# ==========================================================================
def create_md17_figure(md17_data_list, output_dir):
    """
    Grouped bar chart: test MSE per molecule per model (log scale).
    Combines results from multiple experiment runs.
    """
    # Merge results from all runs
    all_results = {}
    for data in md17_data_list:
        for mol, models in data['results'].items():
            all_results[mol] = models

    if not all_results:
        print("WARNING: No MD17 results found, skipping figure")
        return

    molecules = list(all_results.keys())
    model_names = ['hybrid_hcnet', 'clifford_net', 'egnn', 'baseline']

    fig, ax = plt.subplots(figsize=(max(10, len(molecules) * 1.5), 5))

    x = np.arange(len(molecules))
    width = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for j, model_name in enumerate(model_names):
        mse_values = []
        for mol in molecules:
            if mol in all_results and model_name in all_results[mol]:
                mse_values.append(all_results[mol][model_name]['test_mse'])
            else:
                mse_values.append(np.nan)

        bars = ax.bar(x + offsets[j] * width, mse_values, width,
                      label=LABELS.get(model_name, model_name),
                      color=COLORS.get(model_name, '#999999'),
                      edgecolor='black', linewidth=0.6)

    ax.set_ylabel('Test MSE (log scale)', fontsize=13, fontweight='bold')
    ax.set_title('MD17 Force Prediction: Test MSE by Molecule', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in molecules], fontsize=11, rotation=30, ha='right')
    ax.set_yscale('log')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3, which='both')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(output_dir / f'fig_md17_molecules.{ext}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: fig_md17_molecules.pdf')


# ==========================================================================
# Figure 3: Scaling
# ==========================================================================
def create_scaling_figure(scaling_data, output_dir):
    """
    Log-log forward time vs N with power law fits.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    results = scaling_data['results']

    for model_name, model_data in results.items():
        times = model_data['forward_times']
        n_list = sorted([int(k) for k in times.keys()])
        time_list = [times[str(n)] for n in n_list]

        alpha = model_data.get('complexity', {}).get('alpha', 0)
        r2 = model_data.get('complexity', {}).get('r_squared', 0)
        label = f"{LABELS.get(model_name, model_name)} (O(N^{{{alpha:.2f}}}), R²={r2:.2f})"

        ax.loglog(n_list, time_list,
                  marker=MARKERS.get(model_name, 'o'),
                  color=COLORS.get(model_name, '#999999'),
                  linewidth=2, markersize=8,
                  label=label)

    ax.set_xlabel('Number of Particles (N)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Forward Time (ms)', fontsize=13, fontweight='bold')
    ax.set_title('Computational Scaling: Forward Pass Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(output_dir / f'fig_scaling.{ext}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: fig_scaling.pdf')


# ==========================================================================
# Figure 4: OOD Rotation
# ==========================================================================
def create_ood_figure(ood_data, output_dir):
    """
    Bar chart: MSE vs rotation angle per model.
    """
    # ood_data['ood_results'] has structure: {molecule: {model: {angle_str: {mse, mae}}}}
    ood_results = ood_data.get('ood_results', {})
    if not ood_results:
        print("WARNING: No OOD results found, skipping figure")
        return

    # Use first molecule
    molecule = list(ood_results.keys())[0]
    mol_results = ood_results[molecule]

    model_names = list(mol_results.keys())
    # Get angle list from first model
    first_model = model_names[0]
    angles = list(mol_results[first_model].keys())

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(angles))
    width = 0.8 / len(model_names)

    for j, model_name in enumerate(model_names):
        mse_values = [mol_results[model_name][a]['mse'] for a in angles]
        offset = (j - len(model_names)/2 + 0.5) * width
        ax.bar(x + offset, mse_values, width,
               label=LABELS.get(model_name, model_name),
               color=COLORS.get(model_name, '#999999'),
               edgecolor='black', linewidth=0.6)

    # Format angle labels
    angle_labels = []
    for a in angles:
        # Parse "(x, y, z)" format
        a_clean = a.strip('()').replace(' ', '')
        angle_labels.append(f'({a_clean})')

    ax.set_ylabel('Test MSE', fontsize=13, fontweight='bold')
    ax.set_xlabel('Rotation Angles (degrees)', fontsize=13, fontweight='bold')
    ax.set_title(f'OOD Rotation Generalization ({molecule.capitalize()})',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(angle_labels, fontsize=9, rotation=30, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(output_dir / f'fig_ood_rotation.{ext}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: fig_ood_rotation.pdf')


# ==========================================================================
# Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description='Generate figures for Chiral-Global HC-Net paper')
    parser.add_argument('--results_dir', type=str,
                        default=str(Path(__file__).parent.parent / 'results'),
                        help='Results directory')
    parser.add_argument('--output', type=str,
                        default=str(Path(__file__).parent / 'figures'),
                        help='Output directory for figures')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Results dir: {results_dir}')
    print(f'Output dir: {output_dir}')
    print()

    # --- Figure 1: Grade Hierarchy ---
    spiral_file = find_latest('exp1_chirality_spiral_*.json', results_dir)
    rotation_file = find_latest('exp1_chirality_rotation_*.json', results_dir)
    if spiral_file and rotation_file:
        print('Generating grade hierarchy figure...')
        spiral_data = load_json(spiral_file)
        rotation_data = load_json(rotation_file)
        create_grade_hierarchy_figure(spiral_data, rotation_data, output_dir)
    else:
        print(f'SKIP: Grade hierarchy (spiral={spiral_file}, rotation={rotation_file})')

    # --- Figure 2: MD17 Molecules ---
    md17_files = sorted(globmod.glob(str(results_dir / 'exp3_md17_*.json')))
    if md17_files:
        print('Generating MD17 molecules figure...')
        md17_data_list = [load_json(f) for f in md17_files]
        create_md17_figure(md17_data_list, output_dir)
    else:
        print('SKIP: MD17 molecules (no exp3 files found)')

    # --- Figure 3: Scaling ---
    scaling_file = find_latest('exp4_scaling_*.json', results_dir)
    if scaling_file:
        print('Generating scaling figure...')
        scaling_data = load_json(scaling_file)
        create_scaling_figure(scaling_data, output_dir)
    else:
        print('SKIP: Scaling (no exp4 file found)')

    # --- Figure 4: OOD Rotation ---
    # Find exp3 file with ood_results
    ood_file = None
    for f in reversed(md17_files):
        data = load_json(f)
        if 'ood_results' in data and data['ood_results']:
            ood_file = f
            break
    if ood_file:
        print('Generating OOD rotation figure...')
        ood_data = load_json(ood_file)
        create_ood_figure(ood_data, output_dir)
    else:
        print('SKIP: OOD rotation (no ood_results found in exp3 files)')

    print('\nDone!')


if __name__ == '__main__':
    main()
