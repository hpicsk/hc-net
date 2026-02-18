"""
Angular Momentum Classification: Definitive Test of Bivector Importance.

This experiment directly tests whether models can extract global rotation
information from mean-field aggregation.

Task: Classify whether a spinning system rotates CW or CCW

Key Insight:
- Sum of velocity vectors ≈ 0 (cancels due to symmetry)
- Sum of bivectors (r ∧ v) ≠ 0 (all same sign = angular momentum)

Expected Results:
- HC-Net (with bivectors): Should achieve ~100% accuracy
  - Bivector mean-field preserves angular momentum sign
- VectorMeanFieldNet (no bivectors): Should achieve ~50% accuracy (random guess)
  - Vector mean-field loses rotation direction (sum ≈ 0)

This is the DEFINITIVE proof of bivector importance for mean-field.
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

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.spinning_nbody import SpinningNBodySimulator


class AngularMomentumClassificationDataset(Dataset):
    """
    Dataset for angular momentum direction classification.

    Task: Given a spinning N-body system, predict whether it rotates
    clockwise (CW, label=0) or counter-clockwise (CCW, label=1).

    This task REQUIRES global information that vector averaging destroys.
    """

    def __init__(
        self,
        n_samples: int = 5000,
        n_particles: int = 5,
        seed: int = 42
    ):
        self.n_samples = n_samples
        self.n_particles = n_particles

        np.random.seed(seed)

        simulator = SpinningNBodySimulator(n_particles=n_particles)

        self.inputs = []
        self.labels = []

        for i in range(n_samples):
            # Randomly choose direction
            direction = np.random.choice([-1, 1])  # -1 = CW, 1 = CCW
            label = 1 if direction == 1 else 0  # CCW = 1, CW = 0

            # Generate spinning state
            state = simulator.initialize_spinning(
                radius_mean=3.0 + np.random.randn() * 0.5,
                direction=direction,
                seed=seed + i
            )

            # Warmup to get varied configurations
            warmup = np.random.randint(10, 50)
            for _ in range(warmup):
                state = simulator.step(state)

            self.inputs.append(state.to_array())
            self.labels.append(label)

        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

        # Verify data balance
        ccw_count = (self.labels == 1).sum()
        cw_count = (self.labels == 0).sum()
        print(f"  Dataset: {ccw_count} CCW, {cw_count} CW samples")

        # Verify velocity cancellation
        avg_velocity = np.mean(self.inputs[:, :, 2:4], axis=1)  # [n_samples, 2]
        avg_vel_magnitude = np.linalg.norm(avg_velocity, axis=1).mean()
        particle_vel_magnitude = np.linalg.norm(self.inputs[:, :, 2:4], axis=2).mean()
        self.velocity_cancellation = avg_vel_magnitude / (particle_vel_magnitude + 1e-8)
        print(f"  Velocity cancellation ratio: {self.velocity_cancellation:.4f}")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


class ClassifierHead(nn.Module):
    """Classification head that pools particle features."""

    def __init__(self, hidden_dim: int, pool_type: str = 'mean'):
        super().__init__()
        self.pool_type = pool_type
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim] particle features

        Returns:
            [B, 2] class logits
        """
        if self.pool_type == 'mean':
            pooled = x.mean(dim=1)  # [B, dim]
        else:
            pooled = x.max(dim=1)[0]  # [B, dim]

        return self.classifier(pooled)


class HCNetClassifier(nn.Module):
    """HC-Net backbone + classification head."""

    def __init__(self, n_particles: int = 5, hidden_dim: int = 128):
        super().__init__()

        from models.nbody_models import CliffordNBodyNet

        # Use HC-Net backbone (but don't predict next state)
        self.backbone = CliffordNBodyNet(
            n_particles=n_particles,
            hidden_dim=hidden_dim,
            n_layers=4,
            use_attention=False,
            use_mean_field=True
        )
        # Remove output projection, use intermediate features
        self.input_proj = self.backbone.input_proj
        self.layers = self.backbone.layers
        self.mean_field = self.backbone.mean_field
        self.geo_interaction = self.backbone.geo_interaction

        # Classification head
        self.classifier = ClassifierHead(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        h = self.mean_field(h)
        h = self.geo_interaction(h)
        return self.classifier(h)


class VectorMeanFieldClassifier(nn.Module):
    """VectorMeanFieldNet backbone + classification head."""

    def __init__(self, n_particles: int = 5, hidden_dim: int = 128):
        super().__init__()

        from models.vector_meanfield import VectorMeanFieldNet

        # Use VectorMeanFieldNet backbone
        self.backbone = VectorMeanFieldNet(
            n_particles=n_particles,
            hidden_dim=hidden_dim,
            n_layers=4,
            use_mean_field=True
        )
        self.input_proj = self.backbone.input_proj
        self.layers = self.backbone.layers
        self.mean_field = self.backbone.mean_field

        # Classification head
        self.classifier = ClassifierHead(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        h = self.mean_field(h)
        return self.classifier(h)


class HCNetNoBivectorClassifier(nn.Module):
    """HCNetNoBivector backbone + classification head."""

    def __init__(self, n_particles: int = 5, hidden_dim: int = 128):
        super().__init__()

        from models.nbody_models import CliffordNBodyNetNoBivector

        self.backbone = CliffordNBodyNetNoBivector(
            n_particles=n_particles,
            hidden_dim=hidden_dim,
            n_layers=4,
            use_mean_field=True
        )
        self.input_proj = self.backbone.input_proj
        self.layers = self.backbone.layers
        self.mean_field = self.backbone.mean_field

        self.classifier = ClassifierHead(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        h = self.mean_field(h)
        return self.classifier(h)


def create_classifier(model_name: str, n_particles: int = 5, hidden_dim: int = 128):
    """Create classifier by name."""
    if model_name == 'hcnet':
        return HCNetClassifier(n_particles, hidden_dim)
    elif model_name == 'vector_meanfield':
        return VectorMeanFieldClassifier(n_particles, hidden_dim)
    elif model_name == 'hcnet_no_bivector':
        return HCNetNoBivectorClassifier(n_particles, hidden_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_inp, batch_lbl in loader:
        batch_inp = batch_inp.to(device)
        batch_lbl = batch_lbl.to(device)

        optimizer.zero_grad()
        logits = model(batch_inp)
        loss = criterion(logits, batch_lbl)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == batch_lbl).sum().item()
        total += batch_lbl.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_inp, batch_lbl in loader:
            batch_inp = batch_inp.to(device)
            batch_lbl = batch_lbl.to(device)

            logits = model(batch_inp)
            loss = criterion(logits, batch_lbl)

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == batch_lbl).sum().item()
            total += batch_lbl.size(0)

    return total_loss / len(loader), correct / total


def run_experiment(
    model_name: str,
    n_train: int = 5000,
    n_test: int = 1000,
    n_particles: int = 5,
    hidden_dim: int = 128,
    n_epochs: int = 50,
    batch_size: int = 128,
    lr: float = 0.001,
    seed: int = 42,
    device: str = 'cuda'
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Data
    print(f"Creating datasets...")
    train_dataset = AngularMomentumClassificationDataset(n_train, n_particles, seed)
    test_dataset = AngularMomentumClassificationDataset(n_test, n_particles, seed + 10000)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = create_classifier(model_name, n_particles, hidden_dim)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_test_acc = 0.0

    start_time = time.time()

    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

    training_time = time.time() - start_time

    return {
        'model': model_name,
        'n_params': n_params,
        'best_test_acc': best_test_acc,
        'training_time': training_time,
        'velocity_cancellation': train_dataset.velocity_cancellation,
        'seed': seed
    }


def main():
    parser = argparse.ArgumentParser(description='Angular Momentum Classification')
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--n_particles', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='results/angular_momentum')
    args = parser.parse_args()

    models = ['hcnet', 'vector_meanfield', 'hcnet_no_bivector']

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {model: [] for model in models}

    print("=" * 70)
    print("ANGULAR MOMENTUM CLASSIFICATION: Definitive Bivector Test")
    print("=" * 70)
    print(f"\nTask: Classify rotation direction (CW vs CCW)")
    print(f"Hypothesis:")
    print(f"  - HC-Net (bivectors): Should achieve ~100% accuracy")
    print(f"  - VectorMeanFieldNet: Should achieve ~50% (random guess)")
    print(f"\nConfig:")
    print(f"  Training: {args.n_train}, Test: {args.n_test}")
    print(f"  Epochs: {args.n_epochs}, Seeds: {args.seeds}")
    print("=" * 70)

    for seed in args.seeds:
        print(f"\n{'='*70}")
        print(f"SEED: {seed}")
        print(f"{'='*70}")

        for model_name in models:
            print(f"\n--- Training {model_name} ---")

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
                device=args.device
            )

            all_results[model_name].append(result)
            print(f"  Best test accuracy: {result['best_test_acc']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS (mean ± std across seeds)")
    print("=" * 70)

    summary = {}
    for model_name in models:
        accs = [r['best_test_acc'] for r in all_results[model_name]]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        summary[model_name] = {'mean_acc': mean_acc, 'std_acc': std_acc, 'all_acc': accs}
        print(f"{model_name:20s}: Accuracy = {mean_acc:.4f} ± {std_acc:.4f}")

    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    hcnet_acc = summary['hcnet']['mean_acc']
    vector_acc = summary['vector_meanfield']['mean_acc']

    print(f"HC-Net accuracy: {hcnet_acc:.4f}")
    print(f"VectorMeanFieldNet accuracy: {vector_acc:.4f}")

    if hcnet_acc > 0.8 and vector_acc < 0.6:
        print(f"\n✓ HYPOTHESIS CONFIRMED!")
        print(f"  HC-Net successfully classifies rotation direction")
        print(f"  VectorMeanFieldNet fails (near random guess)")
        print(f"  This proves bivectors preserve angular momentum in mean-field")
    elif hcnet_acc > vector_acc + 0.1:
        print(f"\n~ PARTIAL CONFIRMATION")
        print(f"  HC-Net outperforms VectorMeanFieldNet by {(hcnet_acc - vector_acc)*100:.1f}%")
    else:
        print(f"\n? UNEXPECTED RESULTS")
        print(f"  The difference is smaller than expected")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'angular_momentum_{timestamp}.json'

    save_data = {
        'summary': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                        for kk, vv in v.items()} for k, v in summary.items()},
        'all_results': [{k: float(v) if isinstance(v, (np.floating, float)) else v
                         for k, v in r.items()} for model_results in all_results.values()
                        for r in model_results],
        'config': vars(args),
        'timestamp': timestamp
    }

    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return summary


if __name__ == '__main__':
    main()
