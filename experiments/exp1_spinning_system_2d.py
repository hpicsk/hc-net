"""
Experiment 1: Spinning System Classification — Grade Hierarchy Demonstration.

Proves that bivector mean-fields capture rotation information that scalar
and vector mean-fields destroy via averaging.

Task: Binary classification of CW vs CCW spinning particle systems.

Models (all use global mean-field aggregation, differ only in grade):
- ScalarMeanFieldClassifier:  Projects to scalar -> average -> classify
  Expected: ~50% (scalar carries no directional information)
- VectorMeanFieldClassifier:  Projects to vector -> average magnitude -> classify
  Expected: ~50% (symmetric velocities cancel: mean(v) ≈ 0)
- BivectorMeanFieldClassifier: Computes r∧v -> average -> classify
  Expected: ~100% (angular momentum L = sum(r×v) preserves rotation sign)

This is the "killer experiment" showing HC-Net's key insight:
higher Clifford grades preserve geometric information that lower grades destroy.
"""

import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from nips_hcnet.data.spinning_nbody import (
    generate_spinning_nbody_dataset,
)


# ============================================================================
# Classification Dataset
# ============================================================================

class SpinningClassificationDataset(Dataset):
    """
    Spinning N-body dataset with CW/CCW binary labels.

    Each sample is a system of N particles with positions and velocities.
    Label = 1 if CCW (positive angular momentum), 0 if CW (negative).
    """

    def __init__(
        self,
        n_samples: int = 5000,
        n_particles: int = 5,
        seed: int = 42,
    ):
        # Generate with mixed directions (CW and CCW)
        inputs, _, angular_momenta = generate_spinning_nbody_dataset(
            n_trajectories=n_samples,
            n_particles=n_particles,
            prediction_horizon=1,
            seed=seed,
            mixed_directions=True,
        )

        self.inputs = inputs  # [n_samples, n_particles, 4]
        # Label: 1 = CCW (L > 0), 0 = CW (L < 0)
        self.labels = (angular_momenta > 0).astype(np.float32)
        self.angular_momenta = angular_momenta

        # Verify balance
        n_ccw = self.labels.sum()
        n_cw = len(self.labels) - n_ccw
        self.balance = {'ccw': int(n_ccw), 'cw': int(n_cw)}

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


def get_classification_loaders(
    n_train=5000, n_test=1000, n_particles=5, batch_size=128, seed=42
):
    train_ds = SpinningClassificationDataset(n_train, n_particles, seed=seed)
    test_ds = SpinningClassificationDataset(n_test, n_particles, seed=seed + 10000)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    stats = {
        'train_balance': train_ds.balance,
        'test_balance': test_ds.balance,
        'train_mean_abs_L': float(np.abs(train_ds.angular_momenta).mean()),
    }
    return train_loader, test_loader, stats


# ============================================================================
# Mean-Field Classifiers (differ only in aggregation grade)
#
# CRITICAL DESIGN: No learned features BEFORE aggregation.
# The per-particle features are explicit grade projections (no MLPs).
# Only the classifier AFTER mean-pooling is learned.
# This ensures the information bottleneck is genuinely at the aggregation step.
# ============================================================================

class ScalarMeanFieldClassifier(nn.Module):
    """
    Extracts per-particle scalar invariants (||r||², ||v||², r·v),
    averages them across particles, then classifies.

    No learned features before aggregation — only explicit scalar invariants.
    These carry magnitudes but NO rotation direction information.

    Expected: ~50% accuracy (random).
    """

    def __init__(self, hidden_dim=64):
        super().__init__()
        # 3 scalar invariants per particle: ||r||², ||v||², r·v
        # After averaging: 3 features
        self.classifier = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """x: [B, N, 4] -> [B, 1] logits"""
        pos = x[:, :, :2]  # [B, N, 2]
        vel = x[:, :, 2:]  # [B, N, 2]

        # Scalar invariants per particle (no direction info)
        r_sq = (pos ** 2).sum(dim=-1, keepdim=True)       # [B, N, 1]
        v_sq = (vel ** 2).sum(dim=-1, keepdim=True)       # [B, N, 1]
        r_dot_v = (pos * vel).sum(dim=-1, keepdim=True)   # [B, N, 1]

        scalars = torch.cat([r_sq, v_sq, r_dot_v], dim=-1)  # [B, N, 3]

        # Average across particles (scalar mean-field)
        mean_scalars = scalars.mean(dim=1)  # [B, 3]

        return self.classifier(mean_scalars)  # [B, 1]


class VectorMeanFieldClassifier(nn.Module):
    """
    Uses raw position and velocity vectors, averages them across particles,
    then extracts invariant features (magnitudes, dot products) and classifies.

    No learned features before aggregation — raw vectors only.
    For symmetric spinning systems, mean(v) ≈ 0 and mean(r) ≈ 0,
    so the invariants after averaging are near-zero and uninformative.

    Expected: ~50% accuracy (random).
    """

    def __init__(self, hidden_dim=64):
        super().__init__()
        # After averaging: mean_r (2D), mean_v (2D) -> invariants:
        # ||mean_r||², ||mean_v||², mean_r · mean_v = 3 features
        self.classifier = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """x: [B, N, 4] -> [B, 1] logits"""
        pos = x[:, :, :2]  # [B, N, 2]
        vel = x[:, :, 2:]  # [B, N, 2]

        # Average vectors across particles (vector mean-field)
        # For symmetric orbits: mean(r) ≈ 0, mean(v) ≈ 0
        mean_r = pos.mean(dim=1)  # [B, 2]
        mean_v = vel.mean(dim=1)  # [B, 2]

        # Extract invariants from mean vectors
        mr_sq = (mean_r ** 2).sum(dim=-1, keepdim=True)      # [B, 1]
        mv_sq = (mean_v ** 2).sum(dim=-1, keepdim=True)      # [B, 1]
        mr_dot_mv = (mean_r * mean_v).sum(dim=-1, keepdim=True)  # [B, 1]

        invariants = torch.cat([mr_sq, mv_sq, mr_dot_mv], dim=-1)  # [B, 3]

        return self.classifier(invariants)  # [B, 1]


class BivectorMeanFieldClassifier(nn.Module):
    """
    Computes 2D bivector (angular momentum) r∧v for each particle,
    averages over particles, then classifies.

    No learned features before aggregation — explicit wedge product only.
    In 2D, the bivector r∧v = x*vy - y*vx is a pseudoscalar (1 number).
    For spinning systems all particles contribute the same sign,
    so the mean preserves the rotation direction.

    Expected: ~100% accuracy.
    """

    def __init__(self, hidden_dim=64):
        super().__init__()
        # 1 bivector (pseudoscalar) per particle, averaged to 1 feature
        self.classifier = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """x: [B, N, 4] -> [B, 1] logits"""
        pos = x[:, :, :2]  # [B, N, 2]
        vel = x[:, :, 2:]  # [B, N, 2]

        # 2D wedge product: r∧v = x*vy - y*vx (angular momentum per particle)
        bivector = pos[:, :, 0] * vel[:, :, 1] - pos[:, :, 1] * vel[:, :, 0]  # [B, N]

        # Average bivector across particles (bivector mean-field)
        # All same sign → mean preserves rotation direction
        mean_biv = bivector.mean(dim=1, keepdim=True)  # [B, 1]

        return self.classifier(mean_biv)  # [B, 1]


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_inp, batch_lbl in loader:
        batch_inp = batch_inp.to(device)
        batch_lbl = batch_lbl.to(device)

        optimizer.zero_grad()
        logits = model(batch_inp).squeeze(-1)  # [B]
        loss = nn.functional.binary_cross_entropy_with_logits(logits, batch_lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch_inp.size(0)
        preds = (logits > 0).float()
        correct += (preds == batch_lbl).sum().item()
        total += batch_inp.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_inp, batch_lbl in loader:
            batch_inp = batch_inp.to(device)
            batch_lbl = batch_lbl.to(device)

            logits = model(batch_inp).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, batch_lbl)

            total_loss += loss.item() * batch_inp.size(0)
            preds = (logits > 0).float()
            correct += (preds == batch_lbl).sum().item()
            total += batch_inp.size(0)

    return total_loss / total, correct / total


def run_experiment(
    model_name,
    n_train=5000,
    n_test=1000,
    n_particles=5,
    hidden_dim=64,
    n_epochs=100,
    batch_size=128,
    lr=0.001,
    seed=42,
    device='cuda',
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader, stats = get_classification_loaders(
        n_train=n_train, n_test=n_test, n_particles=n_particles,
        batch_size=batch_size, seed=seed,
    )

    # Create model
    if model_name == 'scalar':
        model = ScalarMeanFieldClassifier(hidden_dim)
    elif model_name == 'vector':
        model = VectorMeanFieldClassifier(hidden_dim)
    elif model_name == 'bivector':
        model = BivectorMeanFieldClassifier(hidden_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_test_acc = 0.0
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    start_time = time.time()

    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: "
                  f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

    training_time = time.time() - start_time

    return {
        'model': model_name,
        'n_params': n_params,
        'best_test_acc': best_test_acc,
        'final_train_acc': train_accs[-1],
        'final_test_acc': test_accs[-1],
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
        'train_accs': train_accs,
        'test_accs': test_accs,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'training_time': training_time,
        'dataset_stats': stats,
        'config': {
            'n_train': n_train, 'n_test': n_test, 'n_particles': n_particles,
            'hidden_dim': hidden_dim, 'n_epochs': n_epochs,
            'batch_size': batch_size, 'lr': lr, 'seed': seed,
        }
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Exp1: Spinning System CW/CCW Classification')
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--n_particles', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str,
                        default='results/spinning_system')
    args = parser.parse_args()

    models = ['scalar', 'vector', 'bivector']

    output_dir = Path(__file__).parent.parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {m: [] for m in models}

    print("=" * 70)
    print("EXPERIMENT 1: CW vs CCW CLASSIFICATION — Grade Hierarchy Demo")
    print("=" * 70)
    print()
    print("Hypothesis:")
    print("  Scalar mean-field:   ~50% (no directional info)")
    print("  Vector mean-field:   ~50% (mean(v) cancels for symmetric orbits)")
    print("  Bivector mean-field: ~100% (mean(r^v) = angular momentum, same sign)")
    print()
    print(f"Training: {args.n_train} samples, {args.n_epochs} epochs")
    print(f"Seeds: {args.seeds}")
    print("=" * 70)

    for seed in args.seeds:
        print(f"\n{'='*70}")
        print(f"SEED: {seed}")
        print(f"{'='*70}")

        for model_name in models:
            print(f"\n--- {model_name}_meanfield ---")

            result = run_experiment(
                model_name=model_name,
                n_train=args.n_train,
                n_test=args.n_test,
                n_particles=args.n_particles,
                hidden_dim=args.hidden_dim,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=seed,
                device=args.device,
            )

            all_results[model_name].append(result)
            print(f"  Best test accuracy: {result['best_test_acc']:.4f}")
            print(f"  Training time: {result['training_time']:.1f}s")

    # Aggregate results
    print("\n" + "=" * 70)
    print("FINAL RESULTS (mean +/- std across seeds)")
    print("=" * 70)

    summary = {}
    for model_name in models:
        accs = [r['best_test_acc'] for r in all_results[model_name]]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        summary[model_name] = {
            'mean_acc': float(mean_acc),
            'std_acc': float(std_acc),
            'all_acc': [float(a) for a in accs],
        }
        print(f"  {model_name + '_meanfield':<25s}: "
              f"Accuracy = {mean_acc:.4f} +/- {std_acc:.4f}")

    # Analysis
    scalar_acc = summary['scalar']['mean_acc']
    vector_acc = summary['vector']['mean_acc']
    bivector_acc = summary['bivector']['mean_acc']

    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    print(f"  Scalar mean-field accuracy:   {scalar_acc:.4f} "
          f"({'EXPECTED ~50%' if scalar_acc < 0.7 else 'UNEXPECTED'})")
    print(f"  Vector mean-field accuracy:   {vector_acc:.4f} "
          f"({'EXPECTED ~50%' if vector_acc < 0.7 else 'UNEXPECTED'})")
    print(f"  Bivector mean-field accuracy: {bivector_acc:.4f} "
          f"({'EXPECTED ~100%' if bivector_acc > 0.9 else 'UNEXPECTED'})")

    if bivector_acc > 0.9 and scalar_acc < 0.7 and vector_acc < 0.7:
        print(f"\nHYPOTHESIS CONFIRMED:")
        print(f"  Only bivector (grade 2) mean-field preserves rotation direction.")
        print(f"  Scalar/vector averaging destroys this information.")
        print(f"  This demonstrates the Clifford algebra grade hierarchy:")
        print(f"  scalar (grade 0) < vector (grade 1) < bivector (grade 2)")
    else:
        print(f"\nRESULTS UNEXPECTED — check data generation and models.")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'spinning_classification_{timestamp}.json'

    save_data = {
        'summary': summary,
        'all_results': {k: [{kk: vv for kk, vv in r.items()
                             if kk not in ('train_accs', 'test_accs',
                                           'train_losses', 'test_losses')}
                            for r in v]
                        for k, v in all_results.items()},
        'config': vars(args),
        'timestamp': timestamp,
    }

    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    return summary


if __name__ == '__main__':
    main()
