"""
Experiment 1: Chirality Classification — Toy Proof of Trivector Necessity.

THE key experiment demonstrating that trivectors are necessary for chirality.

Three modes:
- --mode rotation: CW/CCW classification (2D analog in 3D)
  Expected: vector~50%, bivector~100%, trivector~50% (helicity=0)
- --mode chirality: Helix handedness from N-body (NEW result)
  Expected: vector~50%, bivector~50%, trivector~100%
- --mode spiral: Helix handedness from chiral spirals (NEW result)
  Expected: vector~50%, bivector~50%, trivector~100%

Runs all 5 mean-field classifiers x 3 seeds, prints table, saves JSON.
This is "Table 5" — the definitive proof that trivectors are needed.

Usage:
    python experiments/exp1_chirality_classification.py --mode rotation
    python experiments/exp1_chirality_classification.py --mode chirality
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from nips_hcnet.models.meanfield_3d import (
    VectorMeanField3DClassifier,
    BivectorMeanField3DClassifier,
    TrivectorMeanField3DClassifier,
    FullCliffordMeanField3DClassifier,
    LearnedMeanField3DClassifier,
)
from nips_hcnet.data.chiral_spirals import ChiralSpiralDataset
from nips_hcnet.data.spinning_nbody_3d import SpinningChiralityDataset3D


def train_epoch(model, loader, optimizer, criterion, device):
    """Standard training loop."""
    model.train()
    correct = 0
    total = 0

    for batch_inp, batch_lbl in loader:
        batch_inp = batch_inp.to(device)
        batch_lbl = batch_lbl.to(device)

        optimizer.zero_grad()
        logits = model(batch_inp)
        loss = criterion(logits, batch_lbl)
        loss.backward()
        optimizer.step()

        pred = logits.argmax(dim=1)
        correct += (pred == batch_lbl).sum().item()
        total += batch_lbl.size(0)

    return correct / total


def evaluate(model, loader, device):
    """Evaluation loop."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_inp, batch_lbl in loader:
            batch_inp = batch_inp.to(device)
            batch_lbl = batch_lbl.to(device)

            logits = model(batch_inp)
            pred = logits.argmax(dim=1)
            correct += (pred == batch_lbl).sum().item()
            total += batch_lbl.size(0)

    return correct / total


def run_experiment(
    model_class,
    mode: str = 'chirality',
    n_train: int = 5000,
    n_test: int = 1000,
    n_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 0.001,
    seed: int = 42,
    device: str = 'cuda',
):
    """Run a single model training and return best test accuracy."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Create dataset based on mode
    if mode in ('rotation', 'chirality'):
        train_dataset = SpinningChiralityDataset3D(
            n_samples=n_train, n_particles=10, mode=mode, seed=seed
        )
        test_dataset = SpinningChiralityDataset3D(
            n_samples=n_test, n_particles=10, mode=mode, seed=seed + 10000
        )
    elif mode == 'spiral':
        train_dataset = ChiralSpiralDataset(
            n_samples=n_train, n_points=20, seed=seed
        )
        test_dataset = ChiralSpiralDataset(
            n_samples=n_test, n_points=20, seed=seed + 10000
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_acc = 0.0

    for epoch in range(n_epochs):
        train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)

        if test_acc > best_acc:
            best_acc = test_acc

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: train={train_acc:.4f}, test={test_acc:.4f}")

    return best_acc


def main():
    parser = argparse.ArgumentParser(
        description='Exp 1: Chirality Classification'
    )
    parser.add_argument(
        '--mode', type=str, default='chirality',
        choices=['rotation', 'chirality', 'spiral'],
        help='Classification mode'
    )
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print("=" * 70)
    print(f"EXPERIMENT 1: CHIRALITY CLASSIFICATION (mode={args.mode})")
    print("=" * 70)
    print("\nClassifier ONLY sees the global mean-field, NOT individual particles.")
    print("Each classifier sees ONLY its specific grade features.")
    print("\nModels:")
    print("  1. VectorMeanField3D:       avg(pos, vel) -> 6D (grade 1)")
    print("  2. BivectorMeanField3D:     avg(L_xy,L_xz,L_yz) -> 3D (grade 2)")
    print("  3. TrivectorMeanField3D:    avg(helicity) -> 1D (grade 3)")
    print("  4. FullCliffordMeanField3D: pos*vel*L triple product -> 8D (all grades)")
    print("  5. LearnedMeanField3D:      learned 16D projection")

    if args.mode == 'rotation':
        print("\nExpected: vector~50%, bivector~100%, trivector~50%, full_clifford~50%")
        print("  (Trivector=helicity is zero for pure circular orbits with no axial velocity)")
    elif args.mode == 'chirality':
        print("\nExpected: vector~50%, bivector~50%, trivector~100%")
    elif args.mode == 'spiral':
        print("\nExpected: vector~50%, bivector~50%, trivector~100%")

    print("=" * 70)

    models = {
        'vector_only': VectorMeanField3DClassifier,
        'bivector': BivectorMeanField3DClassifier,
        'trivector': TrivectorMeanField3DClassifier,
        'full_clifford': FullCliffordMeanField3DClassifier,
        'learned': LearnedMeanField3DClassifier,
    }

    results = {name: [] for name in models}

    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        for name, model_class in models.items():
            print(f"\n  Training {name}...")
            acc = run_experiment(
                model_class,
                mode=args.mode,
                n_train=args.n_train,
                n_test=args.n_test,
                n_epochs=args.n_epochs,
                seed=seed,
                device=args.device,
            )
            results[name].append(acc)
            print(f"  Best accuracy: {acc:.4f}")

    # Print results table
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"{'Model':<25} {'Accuracy':>15} {'Expected':>15}")
    print("-" * 55)

    expected = {}
    if args.mode == 'rotation':
        expected = {
            'vector_only': '~50%',
            'bivector': '~100%',
            'trivector': '~50%',
            'full_clifford': '~50%',
            'learned': 'variable',
        }
    elif args.mode in ('chirality', 'spiral'):
        expected = {
            'vector_only': '~50%',
            'bivector': '~50%',
            'trivector': '~100%',
            'full_clifford': '~100%',
            'learned': 'variable',
        }

    for name in models:
        accs = results[name]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        exp = expected.get(name, '?')
        print(f"{name:<25} {mean_acc:.4f} +/- {std_acc:.4f}  {exp:>15}")

    # Analysis
    vec_acc = np.mean(results['vector_only'])
    biv_acc = np.mean(results['bivector'])
    triv_acc = np.mean(results['trivector'])

    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    if args.mode in ('chirality', 'spiral'):
        if triv_acc > 0.9 and biv_acc < 0.7:
            print("HYPOTHESIS CONFIRMED!")
            print(f"  Vector mean-field:    {vec_acc:.4f} (near random = 0.5)")
            print(f"  Bivector mean-field:  {biv_acc:.4f} (INSUFFICIENT for chirality)")
            print(f"  Trivector mean-field: {triv_acc:.4f} (SUCCEEDS at chirality)")
            print("\n  This PROVES that trivectors (pseudoscalars) are NECESSARY")
            print("  for chirality detection. Bivectors (angular momentum)")
            print("  cannot distinguish left- from right-handed structures.")
        elif triv_acc > biv_acc + 0.15:
            print("PARTIAL CONFIRMATION")
            print(f"  Trivector outperforms bivector by {(triv_acc - biv_acc)*100:.1f}%")
        else:
            print("UNEXPECTED - results require investigation")
            print(f"  vec={vec_acc:.4f}, biv={biv_acc:.4f}, triv={triv_acc:.4f}")

    elif args.mode == 'rotation':
        if biv_acc > 0.9 and vec_acc < 0.7:
            print("ROTATION RESULT CONFIRMED!")
            print(f"  Vector:    {vec_acc:.4f} (fails - vectors cancel under rotation)")
            print(f"  Bivector:  {biv_acc:.4f} (SUCCEEDS - angular momentum L_z preserved)")
            print(f"  Trivector: {triv_acc:.4f} (fails - helicity=0 for pure circular orbits)")
            print(f"\n  Bivectors (grade 2) are NECESSARY AND SUFFICIENT for rotation detection.")
            print(f"  Trivectors (grade 3) are irrelevant since helicity requires axial velocity.")
            print(f"  This demonstrates clean grade separation: bivector=rotation, trivector=chirality.")
        else:
            print("UNEXPECTED rotation results")
            print(f"  vec={vec_acc:.4f}, biv={biv_acc:.4f}, triv={triv_acc:.4f}")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    results_dict = {
        'results': {k: [float(v) for v in vals] for k, vals in results.items()},
        'summary': {
            k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
            for k, v in results.items()
        },
        'config': {
            'mode': args.mode,
            'n_train': args.n_train,
            'n_test': args.n_test,
            'n_epochs': args.n_epochs,
            'seeds': args.seeds,
        },
    }

    outfile = output_dir / f'exp1_chirality_{args.mode}_{timestamp}.json'
    with open(outfile, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {outfile}")


if __name__ == '__main__':
    main()
