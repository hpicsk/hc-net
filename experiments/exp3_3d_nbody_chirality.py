"""
Experiment 2: 3D N-Body Chirality Classification — Physical System Validation.

Tests the grade hierarchy on physically realistic 3D spinning N-body systems.
Compares mean-field classifiers with the full HybridHCNet3DClassifier and
the existing CliffordNBodyNet3D baseline.

Ablation: hybrid model with/without trivector in the global stream.

Usage:
    python experiments/exp2_3d_nbody_chirality.py --mode chirality
    python experiments/exp2_3d_nbody_chirality.py --mode rotation
"""

import os
import json
import argparse
import time
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
)
from nips_hcnet.models.hybrid_hcnet import HybridHCNet3DClassifier
from nips_hcnet.data.spinning_nbody_3d import SpinningChiralityDataset3D


class CliffordNBodyNet3DClassifier(nn.Module):
    """
    Adapter: wrap CliffordNBodyNet3D for classification.

    Uses global mean pooling of the regression output as classification features.
    """

    def __init__(self, n_particles: int = 5, hidden_dim: int = 128):
        super().__init__()

        from nips_hcnet.models.nbody_models_3d import CliffordNBodyNet3D
        self.backbone = CliffordNBodyNet3D(
            n_particles=n_particles, hidden_dim=hidden_dim, n_layers=4
        )
        self.classifier = nn.Sequential(
            nn.Linear(6, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 6]
        Returns:
            [B, 2] logits
        """
        out = self.backbone(x)  # [B, N, 6]
        pooled = out.mean(dim=1)  # [B, 6]
        return self.classifier(pooled)


class HybridNoTrivectorClassifier(nn.Module):
    """
    Ablation: HybridHCNet3DClassifier but global layer only uses 7 components
    (removes pseudoscalar/trivector from mean-field).
    """

    def __init__(self, hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()

        self.model = HybridHCNet3DClassifier(
            hidden_dim=hidden_dim, n_layers=n_layers,
            k_neighbors=5, cutoff=10.0,
        )
        # Override global layers to use only 7 components (no trivector)
        for gl in self.model.global_layers:
            gl.n_components = 7
            gl.particle_proj = nn.Linear(hidden_dim, 7)
            gl.meanfield_proj = nn.Linear(hidden_dim, 7)
            gl.interaction_proj = nn.Linear(7 * 7, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    correct = 0
    total = 0
    total_loss = 0.0

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
        total_loss += loss.item() * batch_lbl.size(0)

    return correct / total, total_loss / total


def evaluate(model, loader, device):
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
    model_factory,
    mode: str = 'chirality',
    n_train: int = 3000,
    n_test: int = 500,
    n_particles: int = 8,
    n_epochs: int = 80,
    batch_size: int = 64,
    lr: float = 0.001,
    seed: int = 42,
    device: str = 'cuda',
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    train_dataset = SpinningChiralityDataset3D(
        n_samples=n_train, n_particles=n_particles, mode=mode, seed=seed
    )
    test_dataset = SpinningChiralityDataset3D(
        n_samples=n_test, n_particles=n_particles, mode=mode, seed=seed + 10000
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = model_factory().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_acc = 0.0
    t0 = time.time()

    for epoch in range(n_epochs):
        train_acc, train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_acc = evaluate(model, test_loader, device)

        if test_acc > best_acc:
            best_acc = test_acc

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: train={train_acc:.4f}, test={test_acc:.4f}")

    elapsed = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())

    return {
        'best_acc': best_acc,
        'time_s': elapsed,
        'n_params': n_params,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Exp 2: 3D N-Body Chirality'
    )
    parser.add_argument('--mode', type=str, default='chirality',
                        choices=['rotation', 'chirality'])
    parser.add_argument('--n_train', type=int, default=3000)
    parser.add_argument('--n_test', type=int, default=500)
    parser.add_argument('--n_particles', type=int, default=8)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print("=" * 70)
    print(f"EXPERIMENT 2: 3D N-BODY CHIRALITY (mode={args.mode})")
    print("=" * 70)

    n_p = args.n_particles
    hidden = 128

    models = {
        'vector_mf': lambda: VectorMeanField3DClassifier(),
        'bivector_mf': lambda: BivectorMeanField3DClassifier(),
        'trivector_mf': lambda: TrivectorMeanField3DClassifier(),
        'full_clifford_mf': lambda: FullCliffordMeanField3DClassifier(),
        'clifford3d_clf': lambda: CliffordNBodyNet3DClassifier(n_p, hidden),
        'hybrid_hcnet': lambda: HybridHCNet3DClassifier(
            hidden_dim=hidden, n_layers=3, k_neighbors=min(5, n_p - 1), cutoff=10.0
        ),
        'hybrid_no_triv': lambda: HybridNoTrivectorClassifier(hidden, 3),
    }

    results = {name: [] for name in models}

    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        for name, factory in models.items():
            print(f"\n  Training {name}...")
            res = run_experiment(
                factory,
                mode=args.mode,
                n_train=args.n_train,
                n_test=args.n_test,
                n_particles=args.n_particles,
                n_epochs=args.n_epochs,
                seed=seed,
                device=args.device,
            )
            results[name].append(res)
            print(f"  Best acc: {res['best_acc']:.4f}, "
                  f"time: {res['time_s']:.1f}s, "
                  f"params: {res['n_params']:,}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Accuracy':>15} {'Params':>12} {'Time':>10}")
    print("-" * 57)

    for name in models:
        accs = [r['best_acc'] for r in results[name]]
        params = results[name][0]['n_params']
        times = [r['time_s'] for r in results[name]]
        print(f"{name:<20} {np.mean(accs):.4f} +/- {np.std(accs):.4f}"
              f"  {params:>10,}  {np.mean(times):>8.1f}s")

    # Ablation analysis
    if 'hybrid_hcnet' in results and 'hybrid_no_triv' in results:
        hybrid_acc = np.mean([r['best_acc'] for r in results['hybrid_hcnet']])
        notriv_acc = np.mean([r['best_acc'] for r in results['hybrid_no_triv']])
        print(f"\nAblation: Trivector contribution = {(hybrid_acc - notriv_acc)*100:.1f}%")

    # Save
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    save_results = {}
    for name in models:
        save_results[name] = {
            'accs': [r['best_acc'] for r in results[name]],
            'mean_acc': float(np.mean([r['best_acc'] for r in results[name]])),
            'n_params': results[name][0]['n_params'],
        }

    outfile = output_dir / f'exp2_nbody_{args.mode}_{timestamp}.json'
    with open(outfile, 'w') as f:
        json.dump({
            'results': save_results,
            'config': vars(args),
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {outfile}")


if __name__ == '__main__':
    main()
