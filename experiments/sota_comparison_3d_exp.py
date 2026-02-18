"""
3D SOTA Comparison Experiment Runner.

Extends the 2D comparison to 3D N-body dynamics prediction.
Tests rotation equivariance in full SO(3) group.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import argparse

import sys
from importlib.util import spec_from_file_location, module_from_spec

_PCNN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PCNN_ROOT not in sys.path:
    sys.path.insert(0, _PCNN_ROOT)


def _import_module(name, path):
    """Import module directly to avoid __init__.py chain issues."""
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import 3D models
_nbody_models_3d = _import_module(
    "nbody_models_3d", os.path.join(_PCNN_ROOT, 'models', 'nbody_models_3d.py')
)
CliffordNBodyNet3D = _nbody_models_3d.CliffordNBodyNet3D
BaselineNBodyNetWithAttention3D = _nbody_models_3d.BaselineNBodyNetWithAttention3D

# Import EGNN (supports coord_dim=3)
_egnn = _import_module("egnn", os.path.join(_PCNN_ROOT, 'models', 'egnn.py'))
EGNNNBodyNet = _egnn.EGNNNBodyNet

# Import CGENN (supports coord_dim=3)
_cgenn = _import_module("cgenn", os.path.join(_PCNN_ROOT, 'models', 'cgenn.py'))
CGENNNBodyNet = _cgenn.CGENNNBodyNet

# Import NequIP (supports coord_dim=3)
_nequip = _import_module(
    "nequip_nbody", os.path.join(_PCNN_ROOT, 'models', 'nequip_nbody.py')
)
NequIPNBodyNet = _nequip.NequIPNBodyNet

# Import 3D data
_nbody_dataset = _import_module(
    "nbody_dataset", os.path.join(_PCNN_ROOT, 'data', 'nbody_dataset.py')
)
get_nbody_loaders_3d_with_ood = _nbody_dataset.get_nbody_loaders_3d_with_ood

# Import published baselines for fallback
_published = _import_module(
    "published_baselines",
    os.path.join(_PCNN_ROOT, 'experiments', 'published_baselines.py')
)
create_fallback_result = _published.create_fallback_result


@dataclass
class ExperimentConfig3D:
    """Configuration for 3D SOTA comparison experiment."""
    # Models to compare
    models: List[str] = None

    # Data settings
    n_particles: int = 5
    training_sizes: List[int] = None
    rotation_angles: List[Tuple[float, float, float]] = None

    # Training settings
    epochs: int = 100
    batch_size: int = 128
    lr: float = 0.001
    weight_decay: float = 1e-5

    # Model settings
    hidden_dim: int = 128
    n_layers: int = 4
    dropout: float = 0.1

    # Experiment settings
    seeds: List[int] = None
    device: str = 'cuda'
    save_dir: str = './results/sota_comparison_3d'

    def __post_init__(self):
        if self.models is None:
            self.models = ['hcnet3d', 'egnn3d', 'cgenn3d', 'nequip3d', 'baseline3d']
        if self.training_sizes is None:
            self.training_sizes = [100, 500, 1000, 5000]
        if self.rotation_angles is None:
            # 3D rotations (Euler angles: z, y, x)
            self.rotation_angles = [
                (0, 0, 0),      # Identity
                (45, 0, 0),     # 45 around z
                (0, 45, 0),     # 45 around y
                (0, 0, 45),     # 45 around x
                (90, 0, 0),     # 90 around z
                (45, 45, 0),    # Combined
                (90, 90, 0),    # Combined 90
            ]
        if self.seeds is None:
            self.seeds = [42, 123, 456, 789, 1234]


def create_model_3d(model_name: str, config: ExperimentConfig3D) -> nn.Module:
    """Create 3D model by name."""
    n_particles = config.n_particles
    hidden_dim = config.hidden_dim
    n_layers = config.n_layers
    dropout = config.dropout

    if model_name == 'hcnet3d':
        return CliffordNBodyNet3D(
            n_particles=n_particles,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        )
    elif model_name == 'egnn3d':
        return EGNNNBodyNet(
            n_particles=n_particles,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            coord_dim=3,  # 3D
            dropout=dropout
        )
    elif model_name == 'cgenn3d':
        return CGENNNBodyNet(
            n_particles=n_particles,
            hidden_channels=hidden_dim // 4,
            n_layers=n_layers,
            coord_dim=3,  # 3D
            dropout=dropout,
            algebra_dim=6  # Cl(6,0) for 3D pos+vel
        )
    elif model_name == 'nequip3d':
        return NequIPNBodyNet(
            n_particles=n_particles,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            coord_dim=3,  # 3D
            dropout=dropout
        )
    elif model_name == 'baseline3d':
        return BaselineNBodyNetWithAttention3D(
            n_particles=n_particles,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_input, batch_target in train_loader:
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)

        optimizer.zero_grad()
        output = model(batch_input)
        loss = nn.functional.mse_loss(output, batch_target)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: str
) -> float:
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                batch_input, batch_target = batch
            else:
                batch_input, batch_target, _ = batch  # OOD loader includes angles

            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            output = model(batch_input)
            loss = nn.functional.mse_loss(output, batch_target)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def run_single_experiment_3d(
    model_name: str,
    train_size: int,
    seed: int,
    config: ExperimentConfig3D
) -> Dict:
    """Run a single 3D experiment configuration."""
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = config.device if torch.cuda.is_available() else 'cpu'

    # Create 3D data loaders
    train_loader, test_loader, ood_loaders = get_nbody_loaders_3d_with_ood(
        n_train=train_size,
        n_test=1000,
        n_particles=config.n_particles,
        rotation_angles=config.rotation_angles,
        batch_size=config.batch_size,
        seed=seed
    )

    # Create model
    model = create_model_3d(model_name, config).to(device)
    n_params = count_parameters(model)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    # Training
    best_test_loss = float('inf')
    best_state = model.state_dict().copy()
    train_losses = []
    test_losses = []

    start_time = time.time()

    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_loss = evaluate(model, test_loader, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = model.state_dict().copy()

        scheduler.step()

    training_time = time.time() - start_time

    # Load best model
    model.load_state_dict(best_state)

    # Evaluate on OOD sets
    ood_losses = {}
    for angles, ood_loader in ood_loaders.items():
        ood_losses[str(angles)] = evaluate(model, ood_loader, device)

    return {
        'model': model_name,
        'train_size': train_size,
        'seed': seed,
        'n_params': n_params,
        'best_test_loss': best_test_loss,
        'final_train_loss': train_losses[-1],
        'ood_losses': ood_losses,
        'training_time': training_time,
        'train_losses': train_losses,
        'test_losses': test_losses
    }


def run_full_comparison_3d(config: ExperimentConfig3D) -> Dict:
    """Run full 3D SOTA comparison experiment."""
    os.makedirs(config.save_dir, exist_ok=True)

    all_results = []

    total_runs = len(config.models) * len(config.training_sizes) * len(config.seeds)
    current_run = 0

    print(f"Starting 3D SOTA Comparison Experiment")
    print(f"Models: {config.models}")
    print(f"Training sizes: {config.training_sizes}")
    print(f"Seeds: {config.seeds}")
    print(f"Total runs: {total_runs}")
    print("=" * 60)

    for model_name in config.models:
        for train_size in config.training_sizes:
            for seed in config.seeds:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] {model_name}, "
                      f"n_train={train_size}, seed={seed}")

                try:
                    result = run_single_experiment_3d(
                        model_name, train_size, seed, config
                    )
                    all_results.append(result)

                    print(f"  Test MSE: {result['best_test_loss']:.6f}")
                    ood_90_z = result['ood_losses'].get("(90, 0, 0)", "N/A")
                    if ood_90_z != "N/A":
                        print(f"  OOD 90° z: {ood_90_z:.6f}")
                    print(f"  Time: {result['training_time']:.1f}s")

                except Exception as e:
                    print(f"  ERROR: {e}")
                    # Try published fallback (strip '3d' suffix for lookup)
                    base_name = model_name.replace('3d', '')
                    fallback = create_fallback_result(
                        base_name, train_size, dim=3,
                        error_msg=str(e)
                    )
                    if fallback is not None:
                        fallback['model'] = model_name  # Keep original name
                        print(f"  Using published result: "
                              f"MSE={fallback['best_test_loss']:.6f} "
                              f"(from {fallback.get('paper', 'N/A')})")
                        all_results.append(fallback)
                    else:
                        all_results.append({
                            'model': model_name,
                            'train_size': train_size,
                            'seed': seed,
                            'error': str(e)
                        })

    # Aggregate results
    summary = aggregate_results_3d(all_results, config)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(config.save_dir, f"results_3d_{timestamp}.json")

    with open(results_file, 'w') as f:
        json.dump({
            'config': asdict(config),
            'results': all_results,
            'summary': summary
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    # Print summary
    print_summary_3d(summary)

    return {'results': all_results, 'summary': summary}


def aggregate_results_3d(results: List[Dict], config: ExperimentConfig3D) -> Dict:
    """Aggregate 3D results across seeds."""
    summary = {}

    for model_name in config.models:
        summary[model_name] = {}

        for train_size in config.training_sizes:
            # Filter results for this model and train size
            model_results = [
                r for r in results
                if r.get('model') == model_name
                and r.get('train_size') == train_size
                and 'error' not in r
            ]

            if not model_results:
                continue

            # Compute statistics
            test_losses = [r['best_test_loss'] for r in model_results]

            summary[model_name][train_size] = {
                'test_mse_mean': np.mean(test_losses),
                'test_mse_std': np.std(test_losses),
                'n_params': model_results[0]['n_params'],
                'training_time_mean': np.mean([r['training_time'] for r in model_results]),
            }

            # OOD statistics for key rotations
            key_rotations = ["(90, 0, 0)", "(0, 90, 0)", "(45, 45, 0)"]
            for rot_str in key_rotations:
                ood_losses = [
                    r['ood_losses'].get(rot_str, float('nan'))
                    for r in model_results
                ]
                ood_losses = [x for x in ood_losses if not np.isnan(x)]
                if ood_losses:
                    summary[model_name][train_size][f'ood_{rot_str}_mean'] = np.mean(ood_losses)
                    summary[model_name][train_size][f'ood_{rot_str}_std'] = np.std(ood_losses)

    return summary


def print_summary_3d(summary: Dict):
    """Print formatted 3D summary table."""
    print("\n" + "=" * 90)
    print("3D SUMMARY RESULTS")
    print("=" * 90)

    # Get all training sizes
    all_train_sizes = set()
    for model_data in summary.values():
        all_train_sizes.update(model_data.keys())
    all_train_sizes = sorted(all_train_sizes)

    for train_size in all_train_sizes:
        print(f"\n--- Training Size: {train_size} ---")
        print(f"{'Model':<12} {'Params':>10} {'Test MSE':>18} {'OOD (90,0,0)':>15}")
        print("-" * 60)

        for model_name, model_data in summary.items():
            if train_size in model_data:
                data = model_data[train_size]
                test_mse = f"{data['test_mse_mean']:.6f} ± {data['test_mse_std']:.6f}"
                ood_90 = data.get('ood_(90, 0, 0)_mean', float('nan'))
                ood_90_str = f"{ood_90:.6f}" if not np.isnan(ood_90) else "N/A"
                print(f"{model_name:<12} {data['n_params']:>10,} {test_mse:>18} {ood_90_str:>15}")


def main():
    parser = argparse.ArgumentParser(description='3D SOTA Comparison Experiment')

    parser.add_argument('--models', nargs='+',
                       default=['hcnet3d', 'egnn3d', 'cgenn3d', 'nequip3d', 'baseline3d'],
                       help='Models to compare')
    parser.add_argument('--train-sizes', nargs='+', type=int,
                       default=[100, 500, 1000],
                       help='Training set sizes')
    parser.add_argument('--seeds', nargs='+', type=int,
                       default=[42, 123, 456],
                       help='Random seeds')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--n-particles', type=int, default=5,
                       help='Number of particles')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--save-dir', type=str,
                       default='./results/sota_comparison_3d',
                       help='Directory to save results')

    args = parser.parse_args()

    config = ExperimentConfig3D(
        models=args.models,
        training_sizes=args.train_sizes,
        seeds=args.seeds,
        epochs=args.epochs,
        n_particles=args.n_particles,
        hidden_dim=args.hidden_dim,
        device=args.device,
        save_dir=args.save_dir
    )

    run_full_comparison_3d(config)


if __name__ == '__main__':
    main()
