"""
Experiment 4: O(N) Scaling Demonstration.

Measures forward time, backward time, and peak GPU memory as N increases.
Fits O(N^alpha) via log-log regression.

Expected results:
- HybridHCNet3D: alpha ~1.0-1.2 (O(N) from local kNN + global mean-field)
- CliffordNBodyNet3D: alpha ~1.0-1.3 (per-particle processing)
- BaselineNBodyNet3D: alpha ~2.0 (flattened MLP with N*D input)
- EGNN-style: alpha ~2.0 (pairwise interactions)

Usage:
    python experiments/exp4_scaling_analysis.py --device cuda
    python experiments/exp4_scaling_analysis.py --device cpu --n-particles 5 10 20 50
"""

import os
import json
import time
import gc
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import numpy as np

from nips_hcnet.models.hybrid_hcnet import HybridHCNet3D
from nips_hcnet.models.nbody_models_3d import (
    CliffordNBodyNet3D,
    BaselineNBodyNet3D,
)
from nips_hcnet.models.egnn import EGNNNBodyNet


@dataclass
class ScalingConfig:
    """Configuration for scaling experiment."""
    n_particles_list: List[int] = field(
        default_factory=lambda: [10, 50, 100, 500, 1000, 2000, 5000]
    )
    models: List[str] = field(
        default_factory=lambda: [
            'hybrid_hcnet', 'egnn3d', 'clifford3d', 'baseline3d'
        ]
    )
    batch_size: int = 16
    hidden_dim: int = 128
    n_layers: int = 4
    n_warmup: int = 10
    n_iterations: int = 50
    device: str = 'cuda'
    save_dir: str = ''


def create_model(
    model_name: str, n_particles: int, config: ScalingConfig
) -> nn.Module:
    """Create model by name."""
    hidden = config.hidden_dim
    n_layers = config.n_layers

    if model_name == 'hybrid_hcnet':
        return HybridHCNet3D(
            hidden_dim=hidden,
            n_layers=n_layers,
            k_neighbors=min(10, max(1, n_particles - 1)),
            cutoff=10.0,
        )
    elif model_name == 'egnn3d':
        return EGNNNBodyNet(
            n_particles=n_particles,
            hidden_dim=hidden,
            n_layers=n_layers,
            coord_dim=3,
            dropout=0.0,
        )
    elif model_name == 'clifford3d':
        return CliffordNBodyNet3D(
            n_particles=n_particles,
            hidden_dim=hidden,
            n_layers=n_layers,
        )
    elif model_name == 'baseline3d':
        return BaselineNBodyNet3D(
            n_particles=n_particles,
            hidden_dim=min(hidden * 2, 512),
            n_layers=n_layers,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_forward_time(
    model, input_tensor, n_warmup=10, n_iterations=50, device='cuda'
) -> Dict[str, float]:
    model.eval()

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_tensor)

    if device == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_iterations):
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(input_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
    }


def measure_backward_time(
    model, input_tensor, n_warmup=5, n_iterations=20, device='cuda'
) -> Dict[str, float]:
    model.train()

    for _ in range(n_warmup):
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        model.zero_grad()

    if device == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(n_iterations):
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        model.zero_grad()
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
    }


def measure_peak_memory(model, input_tensor, device='cuda') -> Optional[float]:
    if device != 'cuda' or not torch.cuda.is_available():
        return None

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    mem_before = torch.cuda.memory_allocated() / (1024 ** 2)
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)

    mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return mem_peak - mem_before


def compute_complexity_class(timings: Dict[int, float]) -> Dict[str, float]:
    ns = sorted(timings.keys())
    if len(ns) < 3:
        return {'alpha': float('nan'), 'r_squared': float('nan')}

    log_n = np.log(np.array(ns, dtype=float))
    log_t = np.log(np.array([timings[n] for n in ns], dtype=float))

    A = np.vstack([log_n, np.ones_like(log_n)]).T
    result = np.linalg.lstsq(A, log_t, rcond=None)
    alpha, c = result[0]

    ss_res = np.sum((log_t - (alpha * log_n + c)) ** 2)
    ss_tot = np.sum((log_t - np.mean(log_t)) ** 2)
    r_squared = 1 - ss_res / (ss_tot + 1e-12)

    return {'alpha': float(alpha), 'r_squared': float(r_squared)}


def run_scaling_experiment(config: ScalingConfig) -> Dict:
    device = config.device if torch.cuda.is_available() else 'cpu'
    if device != config.device:
        print(f"Warning: {config.device} not available, using {device}")

    results = {}

    print("=" * 70)
    print("SCALING EXPERIMENT: O(N) vs O(N^2)")
    print("=" * 70)
    print(f"Models: {config.models}")
    print(f"N particles: {config.n_particles_list}")
    print(f"Batch size: {config.batch_size}")
    print(f"Device: {device}")
    print()

    for model_name in config.models:
        results[model_name] = {
            'forward_times': {},
            'backward_times': {},
            'peak_memory_mb': {},
            'n_params': {},
            'errors': {},
        }

        print(f"\n--- {model_name} ---")

        for n_particles in config.n_particles_list:
            print(f"  N={n_particles:>4d}: ", end='', flush=True)

            try:
                model = create_model(model_name, n_particles, config).to(device)
                # Adaptive batch size to avoid OOM at large N
                batch_size = config.batch_size
                if n_particles > 2000:
                    batch_size = max(2, batch_size // 8)
                elif n_particles > 1000:
                    batch_size = max(2, batch_size // 4)
                elif n_particles > 500:
                    batch_size = max(4, batch_size // 2)
                input_tensor = torch.randn(
                    batch_size, n_particles, 6, device=device
                )

                n_params = count_parameters(model)
                results[model_name]['n_params'][n_particles] = n_params

                fwd = measure_forward_time(
                    model, input_tensor,
                    n_warmup=config.n_warmup,
                    n_iterations=config.n_iterations,
                    device=device,
                )
                results[model_name]['forward_times'][n_particles] = fwd['mean_ms']

                bwd = measure_backward_time(
                    model, input_tensor,
                    n_warmup=max(config.n_warmup // 2, 2),
                    n_iterations=max(config.n_iterations // 2, 5),
                    device=device,
                )
                results[model_name]['backward_times'][n_particles] = bwd['mean_ms']

                mem = measure_peak_memory(model, input_tensor, device)
                if mem is not None:
                    results[model_name]['peak_memory_mb'][n_particles] = mem

                print(f"fwd={fwd['mean_ms']:>8.2f}ms  "
                      f"bwd={bwd['mean_ms']:>8.2f}ms  "
                      f"params={n_params:>10,}", end='')
                if mem is not None:
                    print(f"  mem={mem:>8.1f}MB", end='')
                print()

                del model, input_tensor
                if device == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                results[model_name]['errors'][n_particles] = str(e)
                print(f"ERROR: {e}")

        fwd_times = results[model_name]['forward_times']
        if len(fwd_times) >= 3:
            complexity = compute_complexity_class(fwd_times)
            results[model_name]['complexity'] = complexity
            print(f"  Fitted complexity: O(N^{complexity['alpha']:.2f}) "
                  f"(R^2={complexity['r_squared']:.3f})")

    print_scaling_summary(results, config)
    return results


def print_scaling_summary(results: Dict, config: ScalingConfig):
    print("\n" + "=" * 70)
    print("SCALING SUMMARY")
    print("=" * 70)

    # Forward time table
    print("\nForward Pass Time (ms) by N:")
    header = f"{'Model':<16}"
    for n in config.n_particles_list:
        header += f"  {'N='+str(n):>10}"
    header += f"  {'O(N^a)':>10}"
    print(header)
    print("-" * len(header))

    for model_name in config.models:
        if model_name not in results:
            continue
        row = f"{model_name:<16}"
        fwd = results[model_name]['forward_times']
        for n in config.n_particles_list:
            if n in fwd:
                row += f"  {fwd[n]:>10.2f}"
            else:
                row += f"  {'ERR':>10}"
        complexity = results[model_name].get('complexity', {})
        alpha = complexity.get('alpha', float('nan'))
        if not np.isnan(alpha):
            row += f"  {'N^%.2f' % alpha:>10}"
        else:
            row += f"  {'N/A':>10}"
        print(row)

    # Complexity classification
    print(f"\nComplexity Classification:")
    print(f"{'Model':<16}  {'Exponent':>10}  {'R^2':>8}  {'Class':>12}")
    print("-" * 50)
    for model_name in config.models:
        if model_name not in results:
            continue
        complexity = results[model_name].get('complexity', {})
        alpha = complexity.get('alpha', float('nan'))
        r2 = complexity.get('r_squared', float('nan'))
        if not np.isnan(alpha):
            if alpha < 1.3:
                cls = 'O(N)'
            elif alpha < 1.7:
                cls = 'O(N log N)'
            else:
                cls = 'O(N^2)'
            print(f"{model_name:<16}  {alpha:>10.2f}  {r2:>8.3f}  {cls:>12}")
        else:
            print(f"{model_name:<16}  {'N/A':>10}  {'N/A':>8}  {'N/A':>12}")

    # Speedup at largest N
    n_list = config.n_particles_list
    if n_list:
        max_n = max(n_list)
        hybrid_time = results.get('hybrid_hcnet', {}).get(
            'forward_times', {}
        ).get(max_n)
        if hybrid_time and hybrid_time > 0:
            print(f"\nSpeedup of hybrid_hcnet at N={max_n}:")
            for model_name in config.models:
                if model_name == 'hybrid_hcnet':
                    continue
                other = results.get(model_name, {}).get(
                    'forward_times', {}
                ).get(max_n)
                if other:
                    ratio = other / hybrid_time
                    print(f"  vs {model_name:<14}: {ratio:.1f}x")


def main():
    parser = argparse.ArgumentParser(description='Exp 4: Scaling Analysis')
    parser.add_argument('--models', nargs='+',
                        default=['hybrid_hcnet', 'egnn3d', 'clifford3d', 'baseline3d'])
    parser.add_argument('--n-particles', nargs='+', type=int,
                        default=[10, 50, 100, 500, 1000, 2000, 5000])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--n-iterations', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ScalingConfig(
        models=args.models,
        n_particles_list=args.n_particles,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        n_iterations=args.n_iterations,
        device=args.device,
        save_dir=str(output_dir),
    )

    results = run_scaling_experiment(config)

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outfile = output_dir / f'exp4_scaling_{timestamp}.json'
    with open(outfile, 'w') as f:
        json.dump({
            'config': asdict(config),
            'results': results,
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {outfile}")


if __name__ == '__main__':
    main()
