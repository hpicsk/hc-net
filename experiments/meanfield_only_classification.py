"""
Mean-Field Only Classification: Direct Test of Information Preservation.

This experiment directly tests whether mean-field aggregation preserves
rotation information by classifying ONLY from the global mean-field state.

Key Design:
- Classifier sees ONLY the averaged global state (not individual particles)
- Vector mean-field: avg(x, y, vx, vy) -> should fail (vectors cancel)
- Bivector mean-field: avg(x, y, vx, vy, x*vy - y*vx) -> should succeed

This is the DEFINITIVE test of the vector averaging collapse hypothesis.
"""

import os
import sys
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

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.spinning_nbody import SpinningNBodySimulator


class SpinningClassificationDataset(Dataset):
    """Dataset for rotation direction classification."""

    def __init__(self, n_samples: int = 5000, n_particles: int = 5, seed: int = 42):
        self.n_samples = n_samples
        np.random.seed(seed)

        simulator = SpinningNBodySimulator(n_particles=n_particles)

        self.inputs = []
        self.labels = []

        for i in range(n_samples):
            direction = np.random.choice([-1, 1])
            label = 1 if direction == 1 else 0

            state = simulator.initialize_spinning(
                radius_mean=3.0 + np.random.randn() * 0.5,
                direction=direction,
                seed=seed + i
            )

            warmup = np.random.randint(10, 50)
            for _ in range(warmup):
                state = simulator.step(state)

            self.inputs.append(state.to_array())
            self.labels.append(label)

        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


class VectorMeanFieldOnlyClassifier(nn.Module):
    """
    Classifier that ONLY uses vector mean-field (no individual particle info).

    Process:
    1. Compute global mean: avg(x, y, vx, vy) over all particles
    2. Classify from this 4D mean vector

    Hypothesis: This will FAIL because sum(vx) ≈ 0 and sum(vy) ≈ 0
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # Mean-field has 4 dims: avg(x), avg(y), avg(vx), avg(vy)
        self.classifier = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 4] particle states

        Returns:
            [B, 2] class logits
        """
        # Compute global mean-field (ONLY this is used for classification)
        mean_field = x.mean(dim=1)  # [B, 4]
        return self.classifier(mean_field)


class BivectorMeanFieldClassifier(nn.Module):
    """
    Classifier that uses bivector-augmented mean-field.

    Process:
    1. Compute for each particle: (x, y, vx, vy, x*vy - y*vx)
    2. Average to get 5D mean-field
    3. Classify from this

    The 5th component (x*vy - y*vx) is the angular momentum, which is
    consistently signed for spinning systems and should enable classification.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # Mean-field has 5 dims: avg(x), avg(y), avg(vx), avg(vy), avg(L)
        self.classifier = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 4] particle states (x, y, vx, vy)

        Returns:
            [B, 2] class logits
        """
        # Extract components
        px = x[:, :, 0]  # [B, N]
        py = x[:, :, 1]  # [B, N]
        vx = x[:, :, 2]  # [B, N]
        vy = x[:, :, 3]  # [B, N]

        # Compute bivector (angular momentum) per particle: L = x*vy - y*vx
        L = px * vy - py * vx  # [B, N]

        # Create augmented features
        augmented = torch.stack([px, py, vx, vy, L], dim=2)  # [B, N, 5]

        # Compute global mean-field
        mean_field = augmented.mean(dim=1)  # [B, 5]

        return self.classifier(mean_field)


class LearnedMeanFieldClassifier(nn.Module):
    """
    Classifier that learns a richer mean-field representation.

    Process:
    1. Project each particle to higher dim
    2. Average to get mean-field
    3. Classify

    This tests whether learned representations can capture rotation info.
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, meanfield_dim: int = 16):
        super().__init__()

        # Per-particle projection
        self.particle_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, meanfield_dim)
        )

        # Classifier from mean-field
        self.classifier = nn.Sequential(
            nn.Linear(meanfield_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project each particle
        h = self.particle_proj(x)  # [B, N, meanfield_dim]

        # Compute mean-field
        mean_field = h.mean(dim=1)  # [B, meanfield_dim]

        return self.classifier(mean_field)


def train_epoch(model, loader, optimizer, criterion, device):
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


def run_experiment(model_class, n_train=5000, n_test=1000, n_epochs=100,
                   batch_size=128, lr=0.001, seed=42, device='cuda'):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    train_dataset = SpinningClassificationDataset(n_train, seed=seed)
    test_dataset = SpinningClassificationDataset(n_test, seed=seed + 10000)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

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
            print(f"  Epoch {epoch+1}: train={train_acc:.4f}, test={test_acc:.4f}")

    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print("=" * 70)
    print("MEAN-FIELD ONLY CLASSIFICATION: Direct Information Test")
    print("=" * 70)
    print("\nClassifier ONLY sees the global mean-field, NOT individual particles.")
    print("\nModels:")
    print("  1. VectorMeanFieldOnly: avg(x, y, vx, vy)")
    print("     -> Should FAIL (vectors cancel: avg(vx) ≈ 0, avg(vy) ≈ 0)")
    print("  2. BivectorMeanField: avg(x, y, vx, vy, x*vy - y*vx)")
    print("     -> Should SUCCEED (avg(L) ≠ 0, encodes rotation)")
    print("  3. LearnedMeanField: learn projection then average")
    print("     -> Can it learn to compute bivector-like features?")
    print("=" * 70)

    models = {
        'vector_only': VectorMeanFieldOnlyClassifier,
        'bivector': BivectorMeanFieldClassifier,
        'learned': LearnedMeanFieldClassifier
    }

    results = {name: [] for name in models}

    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        for name, model_class in models.items():
            print(f"\nTraining {name}...")
            acc = run_experiment(
                model_class,
                n_train=args.n_train,
                n_test=args.n_test,
                n_epochs=args.n_epochs,
                seed=seed,
                device=args.device
            )
            results[name].append(acc)
            print(f"  Best accuracy: {acc:.4f}")

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    for name in models:
        accs = results[name]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"{name:15s}: {mean_acc:.4f} ± {std_acc:.4f}")

    # Analysis
    vec_acc = np.mean(results['vector_only'])
    biv_acc = np.mean(results['bivector'])

    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    if biv_acc > 0.9 and vec_acc < 0.6:
        print("✓ HYPOTHESIS CONFIRMED!")
        print(f"  Vector mean-field: {vec_acc:.4f} (near random = 0.5)")
        print(f"  Bivector mean-field: {biv_acc:.4f} (near perfect)")
        print("\n  This PROVES that vector averaging loses rotation information")
        print("  while bivector (angular momentum) averaging preserves it!")
    elif biv_acc > vec_acc + 0.2:
        print("~ PARTIAL CONFIRMATION")
        print(f"  Bivector outperforms vector by {(biv_acc - vec_acc)*100:.1f}%")
    else:
        print("? UNEXPECTED - results require investigation")

    # Save results
    output_dir = PROJECT_ROOT / 'results' / 'meanfield_only'
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    with open(output_dir / f'meanfield_only_{timestamp}.json', 'w') as f:
        json.dump({
            'results': {k: [float(v) for v in vals] for k, vals in results.items()},
            'summary': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                        for k, v in results.items()},
            'config': vars(args)
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
