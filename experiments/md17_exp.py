"""
MD17 Molecular Dynamics Experiment Runner.

Compares models on the MD17 force prediction benchmark.
Tests on multiple molecules with different sizes.
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
    """Import module directly."""
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import MD17 data
_md17_dataset = _import_module(
    "md17_dataset", os.path.join(_PCNN_ROOT, 'data', 'md17_dataset.py')
)
get_md17_loaders = _md17_dataset.get_md17_loaders
get_md17_loaders_with_ood = _md17_dataset.get_md17_loaders_with_ood
MD17_MOLECULES = _md17_dataset.MD17_MOLECULES

# Import MD17 models
_md17_models = _import_module(
    "md17_models", os.path.join(_PCNN_ROOT, 'models', 'md17_models.py')
)
MD17CliffordNet = _md17_models.MD17CliffordNet
MD17BaselineNet = _md17_models.MD17BaselineNet
MD17EGNNAdapter = _md17_models.MD17EGNNAdapter


@dataclass
class MD17Config:
    """Configuration for MD17 experiments."""
    # Models to compare
    models: List[str] = None

    # Data settings
    molecules: List[str] = None
    n_train: int = 1000
    n_val: int = 500
    n_test: int = 1000

    # Training settings
    epochs: int = 100
    batch_size: int = 32
    lr: float = 0.001
    weight_decay: float = 1e-5

    # Model settings
    hidden_dim: int = 128
    n_layers: int = 4
    dropout: float = 0.1

    # Experiment settings
    seeds: List[int] = None
    device: str = 'cuda'
    save_dir: str = './results/md17'

    def __post_init__(self):
        if self.models is None:
            self.models = ['hcnet', 'egnn', 'baseline']
        if self.molecules is None:
            self.molecules = ['ethanol', 'malonaldehyde', 'uracil']
        if self.seeds is None:
            self.seeds = [42, 123, 456]


def create_md17_model(
    model_name: str,
    n_atoms: int,
    config: MD17Config
) -> nn.Module:
    """Create MD17 model by name."""
    hidden_dim = config.hidden_dim
    n_layers = config.n_layers
    dropout = config.dropout

    if model_name == 'hcnet':
        return MD17CliffordNet(
            n_atoms=n_atoms,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        )
    elif model_name == 'egnn':
        return MD17EGNNAdapter(
            n_atoms=n_atoms,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        )
    elif model_name == 'baseline':
        return MD17BaselineNet(
            n_atoms=n_atoms,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch_md17(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str
) -> float:
    """Train for one epoch on MD17."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for positions, atomic_numbers, forces in train_loader:
        positions = positions.to(device)
        atomic_numbers = atomic_numbers.to(device)
        forces = forces.to(device)

        optimizer.zero_grad()
        pred_forces = model(positions, atomic_numbers)
        loss = nn.functional.mse_loss(pred_forces, forces)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate_md17(
    model: nn.Module,
    test_loader: DataLoader,
    device: str
) -> Tuple[float, float]:
    """
    Evaluate model on MD17.

    Returns:
        (mse, mae) metrics
    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n_samples = 0

    with torch.no_grad():
        for positions, atomic_numbers, forces in test_loader:
            positions = positions.to(device)
            atomic_numbers = atomic_numbers.to(device)
            forces = forces.to(device)

            pred_forces = model(positions, atomic_numbers)

            mse = ((pred_forces - forces) ** 2).mean(dim=-1).mean(dim=-1)
            mae = (pred_forces - forces).abs().mean(dim=-1).mean(dim=-1)

            total_mse += mse.sum().item()
            total_mae += mae.sum().item()
            n_samples += positions.shape[0]

    return total_mse / n_samples, total_mae / n_samples


def run_single_md17_experiment(
    model_name: str,
    molecule: str,
    seed: int,
    config: MD17Config
) -> Dict:
    """Run a single MD17 experiment."""
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = config.device if torch.cuda.is_available() else 'cpu'

    # Get molecule info
    n_atoms = MD17_MOLECULES[molecule]['n_atoms']

    # Create data loaders
    train_loader, val_loader, test_loader = get_md17_loaders(
        molecule=molecule,
        n_train=config.n_train,
        n_val=config.n_val,
        n_test=config.n_test,
        batch_size=config.batch_size,
        seed=seed
    )

    # Create model
    model = create_md17_model(model_name, n_atoms, config).to(device)
    n_params = count_parameters(model)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    # Training
    best_val_mse = float('inf')
    train_losses = []
    val_mses = []

    start_time = time.time()

    for epoch in range(config.epochs):
        train_loss = train_epoch_md17(model, train_loader, optimizer, device)
        val_mse, val_mae = evaluate_md17(model, val_loader, device)

        train_losses.append(train_loss)
        val_mses.append(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = model.state_dict().copy()

        scheduler.step()

    training_time = time.time() - start_time

    # Load best model and evaluate on test
    model.load_state_dict(best_state)
    test_mse, test_mae = evaluate_md17(model, test_loader, device)

    return {
        'model': model_name,
        'molecule': molecule,
        'seed': seed,
        'n_atoms': n_atoms,
        'n_params': n_params,
        'best_val_mse': best_val_mse,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'training_time': training_time,
        'train_losses': train_losses,
        'val_mses': val_mses
    }


def run_full_md17_experiment(config: MD17Config) -> Dict:
    """Run full MD17 comparison experiment."""
    os.makedirs(config.save_dir, exist_ok=True)

    all_results = []

    total_runs = len(config.models) * len(config.molecules) * len(config.seeds)
    current_run = 0

    print(f"Starting MD17 Experiment")
    print(f"Models: {config.models}")
    print(f"Molecules: {config.molecules}")
    print(f"Seeds: {config.seeds}")
    print(f"Total runs: {total_runs}")
    print("=" * 60)

    for model_name in config.models:
        for molecule in config.molecules:
            for seed in config.seeds:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] {model_name}, {molecule}, seed={seed}")

                try:
                    result = run_single_md17_experiment(
                        model_name, molecule, seed, config
                    )
                    all_results.append(result)

                    print(f"  Test MSE: {result['test_mse']:.6f}")
                    print(f"  Test MAE: {result['test_mae']:.6f}")
                    print(f"  Time: {result['training_time']:.1f}s")

                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        'model': model_name,
                        'molecule': molecule,
                        'seed': seed,
                        'error': str(e)
                    })

    # Aggregate results
    summary = aggregate_md17_results(all_results, config)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(config.save_dir, f"results_md17_{timestamp}.json")

    with open(results_file, 'w') as f:
        json.dump({
            'config': asdict(config),
            'results': all_results,
            'summary': summary
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    # Print summary
    print_md17_summary(summary)

    return {'results': all_results, 'summary': summary}


def aggregate_md17_results(results: List[Dict], config: MD17Config) -> Dict:
    """Aggregate MD17 results across seeds."""
    summary = {}

    for model_name in config.models:
        summary[model_name] = {}

        for molecule in config.molecules:
            model_results = [
                r for r in results
                if r.get('model') == model_name
                and r.get('molecule') == molecule
                and 'error' not in r
            ]

            if not model_results:
                continue

            test_mses = [r['test_mse'] for r in model_results]
            test_maes = [r['test_mae'] for r in model_results]

            summary[model_name][molecule] = {
                'test_mse_mean': np.mean(test_mses),
                'test_mse_std': np.std(test_mses),
                'test_mae_mean': np.mean(test_maes),
                'test_mae_std': np.std(test_maes),
                'n_atoms': model_results[0]['n_atoms'],
                'n_params': model_results[0]['n_params'],
                'training_time_mean': np.mean([r['training_time'] for r in model_results])
            }

    return summary


def print_md17_summary(summary: Dict):
    """Print formatted MD17 summary."""
    print("\n" + "=" * 80)
    print("MD17 SUMMARY RESULTS")
    print("=" * 80)

    # Get all molecules
    all_molecules = set()
    for model_data in summary.values():
        all_molecules.update(model_data.keys())
    all_molecules = sorted(all_molecules)

    for molecule in all_molecules:
        print(f"\n--- {molecule.upper()} ---")
        print(f"{'Model':<12} {'Params':>10} {'Test MSE':>20} {'Test MAE':>20}")
        print("-" * 65)

        for model_name, model_data in summary.items():
            if molecule in model_data:
                data = model_data[molecule]
                mse_str = f"{data['test_mse_mean']:.6f} ± {data['test_mse_std']:.6f}"
                mae_str = f"{data['test_mae_mean']:.6f} ± {data['test_mae_std']:.6f}"
                print(f"{model_name:<12} {data['n_params']:>10,} {mse_str:>20} {mae_str:>20}")


def main():
    parser = argparse.ArgumentParser(description='MD17 Experiment')

    parser.add_argument('--models', nargs='+',
                       default=['hcnet', 'egnn', 'baseline'],
                       help='Models to compare')
    parser.add_argument('--molecules', nargs='+',
                       default=['ethanol', 'malonaldehyde'],
                       help='Molecules to test')
    parser.add_argument('--n-train', type=int, default=1000,
                       help='Training samples')
    parser.add_argument('--seeds', nargs='+', type=int,
                       default=[42, 123, 456],
                       help='Random seeds')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--save-dir', type=str,
                       default='./results/md17',
                       help='Directory to save results')

    args = parser.parse_args()

    config = MD17Config(
        models=args.models,
        molecules=args.molecules,
        n_train=args.n_train,
        seeds=args.seeds,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        device=args.device,
        save_dir=args.save_dir
    )

    run_full_md17_experiment(config)


if __name__ == '__main__':
    main()
