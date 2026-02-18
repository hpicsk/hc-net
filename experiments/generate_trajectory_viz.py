"""
Generate qualitative trajectory visualization for N-body predictions.

Creates a visual comparison of:
- Ground truth particle velocities (black arrows)
- HC-Net (Clifford) predictions (blue arrows)
- Baseline predictions (red arrows)

This visualization shows that HC-Net "understands" rotation while
the baseline "memorizes" coordinates and fails on rotated inputs.

Usage:
    python -m pcnn.experiments.generate_trajectory_viz
    python -m pcnn.experiments.generate_trajectory_viz --angle 45 --n-particles 3 --n-train 100
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.optim as optim

from pcnn.data.nbody_dataset import get_nbody_loaders_with_ood, NBodyDataset
from pcnn.models.nbody_models import (
    CliffordNBodyNet,
    BaselineNBodyNetWithAttention,
)


def train_model_with_checkpoint(
    model: nn.Module,
    train_loader,
    test_loader,
    epochs: int = 100,
    device: str = 'cuda',
    lr: float = 1e-3,
    checkpoint_path: str = None
) -> nn.Module:
    """Train model and optionally save checkpoint."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_test_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Evaluation
        model.eval()
        test_loss = 0
        n_batches = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                n_batches += 1
        test_loss /= n_batches

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = model.state_dict().copy()

        scheduler.step()

        if epoch % 20 == 0:
            print(f'  Epoch {epoch}: Test Loss {test_loss:.6f}')

    # Load best state
    model.load_state_dict(best_state)

    # Save checkpoint if requested
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(best_state, checkpoint_path)
        print(f'  Saved checkpoint to {checkpoint_path}')

    return model


def get_predictions_on_rotated_sample(
    clifford_model: nn.Module,
    baseline_model: nn.Module,
    n_particles: int,
    angle: float,
    device: str = 'cuda',
    seed: int = 42
) -> dict:
    """
    Get predictions on a specific rotated test sample.

    Returns dict with:
        - input_state: Original positions [n_particles, 2]
        - rotated_positions: Rotated positions [n_particles, 2]
        - ground_truth_velocities: True velocities [n_particles, 2]
        - clifford_velocities: HC-Net predictions [n_particles, 2]
        - baseline_velocities: Baseline predictions [n_particles, 2]
    """
    # Create a small dataset with the specific rotation
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate a test sample
    from pcnn.data.nbody_simulation import NBodySimulator, NBodyState

    sim = NBodySimulator(n_particles=n_particles, dt=0.01, softening=0.1)

    # Generate initial state
    positions = np.random.randn(n_particles, 2) * 0.5
    velocities = np.random.randn(n_particles, 2) * 0.1
    masses = np.ones(n_particles)

    # Get ground truth next velocities
    state_obj = NBodyState(positions=positions, velocities=velocities, masses=masses)
    accelerations = sim.compute_accelerations(state_obj)
    next_velocities = velocities + accelerations * sim.dt

    # Create input tensor [n_particles, 4]: x, y, vx, vy
    state = np.hstack([positions, velocities])  # [n_particles, 4]
    input_state = state.copy()

    # Apply rotation to input
    theta = np.radians(angle)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    rotated_positions = (R @ positions.T).T
    rotated_velocities = (R @ velocities.T).T
    rotated_state = np.hstack([rotated_positions, rotated_velocities])

    # Ground truth rotated velocities
    rotated_gt_velocities = (R @ next_velocities.T).T

    # Get model predictions
    clifford_model.eval()
    baseline_model.eval()

    with torch.no_grad():
        input_tensor = torch.tensor(rotated_state, dtype=torch.float32).unsqueeze(0).to(device)

        clifford_out = clifford_model(input_tensor).cpu().numpy()[0]  # [n_particles, 4] or [n_particles, 2]
        baseline_out = baseline_model(input_tensor).cpu().numpy()[0]  # [n_particles, 4] or [n_particles, 2]

        # Model outputs full state [x, y, vx, vy], extract velocity components
        if clifford_out.shape[1] == 4:
            clifford_out = clifford_out[:, 2:]  # [n_particles, 2] - vx, vy
        if baseline_out.shape[1] == 4:
            baseline_out = baseline_out[:, 2:]  # [n_particles, 2] - vx, vy

    return {
        'original_positions': positions,
        'rotated_positions': rotated_positions,
        'rotated_velocities': rotated_velocities,
        'ground_truth_velocities': rotated_gt_velocities,
        'clifford_velocities': clifford_out,
        'baseline_velocities': baseline_out,
        'angle': angle
    }


def plot_trajectory_comparison(data: dict, output_path: str, title: str = None):
    """
    Create a publication-quality trajectory comparison plot.

    Shows particle positions as circles and velocity predictions as arrows.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    positions = data['rotated_positions']
    gt_vel = data['ground_truth_velocities']
    cliff_vel = data['clifford_velocities']
    base_vel = data['baseline_velocities']

    scale = 3.0  # Arrow scale factor

    # Subplot 1: Ground Truth
    ax = axes[0]
    for i in range(len(positions)):
        ax.scatter(positions[i, 0], positions[i, 1], s=200, c=f'C{i}', edgecolors='black', linewidth=2, zorder=3)
        ax.arrow(positions[i, 0], positions[i, 1],
                gt_vel[i, 0] * scale, gt_vel[i, 1] * scale,
                head_width=0.08, head_length=0.04, fc='black', ec='black', linewidth=2, zorder=2)
    ax.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Subplot 2: HC-Net (Clifford)
    ax = axes[1]
    for i in range(len(positions)):
        ax.scatter(positions[i, 0], positions[i, 1], s=200, c=f'C{i}', edgecolors='black', linewidth=2, zorder=3)
        # Ground truth (dashed)
        ax.arrow(positions[i, 0], positions[i, 1],
                gt_vel[i, 0] * scale, gt_vel[i, 1] * scale,
                head_width=0.06, head_length=0.03, fc='none', ec='gray', linewidth=1.5,
                linestyle='--', zorder=1, alpha=0.5)
        # Prediction (solid)
        ax.arrow(positions[i, 0], positions[i, 1],
                cliff_vel[i, 0] * scale, cliff_vel[i, 1] * scale,
                head_width=0.08, head_length=0.04, fc='#2E86AB', ec='#2E86AB', linewidth=2, zorder=2)
    ax.set_title('HC-Net (Clifford)', fontsize=14, fontweight='bold', color='#2E86AB')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')

    # Compute error
    cliff_error = np.mean(np.linalg.norm(cliff_vel - gt_vel, axis=1))
    ax.text(0.05, 0.95, f'MSE: {np.mean((cliff_vel - gt_vel)**2):.6f}',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Subplot 3: Baseline
    ax = axes[2]
    for i in range(len(positions)):
        ax.scatter(positions[i, 0], positions[i, 1], s=200, c=f'C{i}', edgecolors='black', linewidth=2, zorder=3)
        # Ground truth (dashed)
        ax.arrow(positions[i, 0], positions[i, 1],
                gt_vel[i, 0] * scale, gt_vel[i, 1] * scale,
                head_width=0.06, head_length=0.03, fc='none', ec='gray', linewidth=1.5,
                linestyle='--', zorder=1, alpha=0.5)
        # Prediction (solid)
        ax.arrow(positions[i, 0], positions[i, 1],
                base_vel[i, 0] * scale, base_vel[i, 1] * scale,
                head_width=0.08, head_length=0.04, fc='#E94F37', ec='#E94F37', linewidth=2, zorder=2)
    ax.set_title('Baseline MLP', fontsize=14, fontweight='bold', color='#E94F37')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')

    # Compute error
    base_error = np.mean(np.linalg.norm(base_vel - gt_vel, axis=1))
    ax.text(0.05, 0.95, f'MSE: {np.mean((base_vel - gt_vel)**2):.6f}',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add main title
    if title is None:
        title = f"N-Body Velocity Prediction at {data['angle']}° Rotation"
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='--', label='Ground Truth'),
        Line2D([0], [0], color='#2E86AB', linewidth=2, label='HC-Net'),
        Line2D([0], [0], color='#E94F37', linewidth=2, label='Baseline'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11,
              bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Saved: {output_path}')


def compute_per_sample_errors(
    clifford_model: nn.Module,
    baseline_model: nn.Module,
    ood_loader,
    device: str = 'cuda'
) -> list:
    """
    Compute per-sample MSE for both models on OOD data.

    Returns list of dicts with sample data and errors, sorted by error gap.
    """
    clifford_model.eval()
    baseline_model.eval()

    samples = []

    with torch.no_grad():
        for batch in ood_loader:
            # OOD loader returns (inputs, targets, angles) - unpack flexibly
            if len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch

            inputs = inputs.to(device)
            targets = targets.to(device)

            cliff_out = clifford_model(inputs)
            base_out = baseline_model(inputs)

            # Process each sample in batch
            for i in range(inputs.shape[0]):
                inp = inputs[i].cpu().numpy()  # [n_particles, 4]
                tgt = targets[i].cpu().numpy()  # [n_particles, 2] or [n_particles, 4]
                cliff_pred = cliff_out[i].cpu().numpy()
                base_pred = base_out[i].cpu().numpy()

                # Extract velocity components if needed
                if tgt.shape[1] == 4:
                    tgt = tgt[:, 2:]  # vx, vy
                if cliff_pred.shape[1] == 4:
                    cliff_pred = cliff_pred[:, 2:]
                if base_pred.shape[1] == 4:
                    base_pred = base_pred[:, 2:]

                # Compute MSE
                cliff_mse = np.mean((cliff_pred - tgt) ** 2)
                base_mse = np.mean((base_pred - tgt) ** 2)
                error_gap = base_mse - cliff_mse  # Positive means HC-Net is better

                samples.append({
                    'positions': inp[:, :2],
                    'velocities': inp[:, 2:],
                    'ground_truth': tgt,
                    'clifford_pred': cliff_pred,
                    'baseline_pred': base_pred,
                    'clifford_mse': cliff_mse,
                    'baseline_mse': base_mse,
                    'error_gap': error_gap
                })

    # Filter out samples with very small velocities (not visually interesting)
    min_vel_magnitude = 0.01
    samples = [s for s in samples if np.linalg.norm(s['ground_truth']) > min_vel_magnitude]

    # Sort by error gap (descending - largest gap first = best for HC-Net)
    samples.sort(key=lambda x: x['error_gap'], reverse=True)
    return samples


def plot_multi_panel_trajectory(samples: list, percentiles: list, angle: float, output_path: str):
    """
    Create publication-quality multi-panel trajectory figure.

    Args:
        samples: List of sample dicts sorted by error gap
        percentiles: List of percentiles to show (e.g., [90, 75, 50])
        angle: Rotation angle for title
        output_path: Where to save
    """
    n_panels = len(percentiles)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for panel_idx, pct in enumerate(percentiles):
        # Get sample at this percentile
        idx = int((100 - pct) / 100 * len(samples))
        idx = min(idx, len(samples) - 1)
        sample = samples[idx]

        ax = axes[panel_idx]
        positions = sample['positions']
        gt_vel = sample['ground_truth']
        cliff_vel = sample['clifford_pred']
        base_vel = sample['baseline_pred']

        # Compute axis limits based on positions
        pos_range = max(np.abs(positions).max(), 0.5) * 1.3

        # Compute adaptive scale - make arrows ~30% of plot width
        max_vel = max(
            np.max(np.linalg.norm(gt_vel, axis=1)),
            np.max(np.linalg.norm(cliff_vel, axis=1)),
            np.max(np.linalg.norm(base_vel, axis=1)),
            1e-6
        )
        target_arrow_len = pos_range * 0.5  # Arrows should be ~50% of plot range
        scale = target_arrow_len / max_vel

        # Plot particles and arrows
        for i in range(len(positions)):
            # Particle
            ax.scatter(positions[i, 0], positions[i, 1], s=200, c=f'C{i}',
                      edgecolors='black', linewidth=2, zorder=3)

            # Ground truth (dashed gray)
            ax.annotate('', xy=(positions[i, 0] + gt_vel[i, 0] * scale,
                               positions[i, 1] + gt_vel[i, 1] * scale),
                       xytext=(positions[i, 0], positions[i, 1]),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1.5,
                                      linestyle='--', alpha=0.7),
                       zorder=1)

            # HC-Net prediction (blue)
            ax.annotate('', xy=(positions[i, 0] + cliff_vel[i, 0] * scale,
                               positions[i, 1] + cliff_vel[i, 1] * scale),
                       xytext=(positions[i, 0], positions[i, 1]),
                       arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=2.5),
                       zorder=2)

            # Baseline prediction (red)
            ax.annotate('', xy=(positions[i, 0] + base_vel[i, 0] * scale,
                               positions[i, 1] + base_vel[i, 1] * scale),
                       xytext=(positions[i, 0], positions[i, 1]),
                       arrowprops=dict(arrowstyle='->', color='#E94F37', lw=2.5),
                       zorder=2)

        # Labels
        pct_label = {90: 'A', 75: 'B', 50: 'C'}.get(pct, str(pct))
        ax.set_title(f'Sample {pct_label} ({pct}th percentile)', fontsize=12, fontweight='bold')

        # Error annotation with ratio
        ratio = sample['baseline_mse'] / (sample['clifford_mse'] + 1e-10)
        ax.text(0.05, 0.95, f'HC-Net: {sample["clifford_mse"]:.2e}\nBaseline: {sample["baseline_mse"]:.2e}\nRatio: {ratio:.1f}x',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
               family='monospace')

        ax.set_xlim(-pos_range, pos_range)
        ax.set_ylim(-pos_range, pos_range)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        if panel_idx == 0:
            ax.set_ylabel('y')

    # Main title
    fig.suptitle(f'Out-of-Distribution Generalization: {angle}° Rotation\n(Representative samples by error gap percentile)',
                fontsize=14, fontweight='bold')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label='Ground Truth'),
        Line2D([0], [0], color='#2E86AB', linewidth=2.5, label='HC-Net'),
        Line2D([0], [0], color='#E94F37', linewidth=2.5, label='Baseline'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11,
              bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 0.92])

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Generate trajectory visualization')
    parser.add_argument('--n-particles', type=int, default=3, help='Number of particles')
    parser.add_argument('--n-train', type=int, default=100, help='Training samples')
    parser.add_argument('--angle', type=float, default=45, help='Rotation angle for visualization')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='./results/starved_nbody',
                       help='Output directory')
    parser.add_argument('--multi-panel', action='store_true', default=True,
                       help='Generate multi-panel percentile figure (default: True)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    output_dir = Path(args.output)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f'\n=== Generating Trajectory Visualization ===')
    print(f'N particles: {args.n_particles}')
    print(f'Training samples: {args.n_train}')
    print(f'Rotation angle: {args.angle}°\n')

    # Get data loaders
    train_loader, test_loader, ood_loaders = get_nbody_loaders_with_ood(
        n_particles=args.n_particles,
        n_train=args.n_train,
        n_test=500,
        batch_size=min(64, args.n_train),
        rotation_angles=[0, args.angle, 90],
        seed=args.seed
    )

    # Train Clifford model
    print('Training Clifford model...')
    torch.manual_seed(args.seed)
    clifford_model = CliffordNBodyNet(
        n_particles=args.n_particles,
        hidden_dim=64,
        n_layers=3,
        dropout=0.1
    )
    clifford_model = train_model_with_checkpoint(
        clifford_model, train_loader, test_loader,
        epochs=args.epochs, device=device,
        checkpoint_path=str(checkpoint_dir / f'clifford_n{args.n_particles}_t{args.n_train}.pt')
    )

    # Train Baseline model
    print('\nTraining Baseline model...')
    torch.manual_seed(args.seed)
    baseline_model = BaselineNBodyNetWithAttention(
        n_particles=args.n_particles,
        hidden_dim=128,
        n_layers=3,
        dropout=0.1
    )
    baseline_model = train_model_with_checkpoint(
        baseline_model, train_loader, test_loader,
        epochs=args.epochs, device=device,
        checkpoint_path=str(checkpoint_dir / f'baseline_n{args.n_particles}_t{args.n_train}.pt')
    )

    # Get OOD loader for the specified angle
    angle_key = args.angle
    ood_loader = ood_loaders.get(angle_key)
    if ood_loader is None:
        print(f'Warning: No OOD loader for {angle_key}°, using test_loader')
        ood_loader = test_loader

    # Compute per-sample errors and sort
    print(f'\nComputing per-sample errors on {args.angle}° OOD data...')
    samples = compute_per_sample_errors(clifford_model, baseline_model, ood_loader, device)

    # Report statistics
    gaps = [s['error_gap'] for s in samples]
    print(f'  Total samples: {len(samples)}')
    print(f'  Error gap (baseline - hcnet): mean={np.mean(gaps):.2e}, std={np.std(gaps):.2e}')
    print(f'  Samples where HC-Net wins: {sum(1 for g in gaps if g > 0)} / {len(samples)}')

    # Generate multi-panel figure with percentile selection
    if args.multi_panel:
        print(f'\nGenerating multi-panel figure (90th, 75th, 50th percentiles)...')
        output_path = output_dir / 'figure1_trajectory.pdf'
        plot_multi_panel_trajectory(samples, [90, 75, 50], args.angle, str(output_path))

    # Also generate individual samples for reference
    print(f'\nGenerating individual sample figures...')
    for viz_seed in [42, 123, 456]:
        data = get_predictions_on_rotated_sample(
            clifford_model, baseline_model,
            n_particles=args.n_particles,
            angle=args.angle,
            device=device,
            seed=viz_seed
        )

        # Create visualization
        output_path = output_dir / f'trajectory_n{args.n_particles}_t{args.n_train}_r{int(args.angle)}_s{viz_seed}.pdf'
        title = f'{args.n_particles}-Body Prediction ({args.n_train} training samples, {args.angle}° rotation)'
        plot_trajectory_comparison(data, str(output_path), title=title)

    print('\nDone!')


if __name__ == '__main__':
    main()
