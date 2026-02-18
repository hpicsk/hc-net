"""
Computational Scaling Experiment: O(N) vs O(N^2).

Demonstrates HC-Net's O(N) scaling advantage over O(N^2) baselines
(EGNN, CGENN, NequIP) by measuring wall-clock time and memory as
the number of particles N increases.

Key claim: HC-Net processes particles with per-particle operations
(O(N) total), while EGNN/CGENN/NequIP compute pairwise interactions
(O(N^2) total). This matters as N grows.

Usage:
    python experiments/scaling_exp.py --device cuda
    python experiments/scaling_exp.py --device cpu --n-particles 5 10 20 50
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import os
import gc
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
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


# Import models
_nbody_models = _import_module(
    "nbody_models", os.path.join(_PCNN_ROOT, 'models', 'nbody_models.py')
)
CliffordNBodyNet = _nbody_models.CliffordNBodyNet
BaselineNBodyNetWithAttention = _nbody_models.BaselineNBodyNetWithAttention

_egnn = _import_module("egnn", os.path.join(_PCNN_ROOT, 'models', 'egnn.py'))
EGNNNBodyNet = _egnn.EGNNNBodyNet

_cgenn = _import_module("cgenn", os.path.join(_PCNN_ROOT, 'models', 'cgenn.py'))
CGENNNBodyNet = _cgenn.CGENNNBodyNet

_nequip = _import_module(
    "nequip_nbody", os.path.join(_PCNN_ROOT, 'models', 'nequip_nbody.py')
)
NequIPNBodyNet = _nequip.NequIPNBodyNet


@dataclass
class ScalingConfig:
    """Configuration for scaling experiment."""
    n_particles_list: List[int] = field(
        default_factory=lambda: [5, 10, 20, 50, 100]
    )
    models: List[str] = field(
        default_factory=lambda: [
            'hcnet', 'hcnet_meanfield', 'hcnet_attn', 'egnn', 'cgenn', 'nequip', 'baseline'
        ]
    )
    batch_size: int = 16
    hidden_dim: int = 128
    n_layers: int = 4
    n_warmup: int = 10
    n_iterations: int = 50
    device: str = 'cuda'
    save_dir: str = './results/scaling'


def create_model(model_name: str, n_particles: int, config: ScalingConfig) -> nn.Module:
    """Create model by name."""
    hidden_dim = config.hidden_dim
    n_layers = config.n_layers

    if model_name == 'hcnet':
        return CliffordNBodyNet(
            n_particles=n_particles,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            use_attention=False,
            use_mean_field=False  # Original: no inter-particle communication
        )
    elif model_name == 'hcnet_meanfield':
        return CliffordNBodyNet(
            n_particles=n_particles,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            use_attention=False,
            use_mean_field=True  # O(N) mean-field inter-particle communication
        )
    elif model_name == 'hcnet_attn':
        return CliffordNBodyNet(
            n_particles=n_particles,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            use_attention=True,
            use_mean_field=False  # Attention already provides interaction
        )
    elif model_name == 'egnn':
        return EGNNNBodyNet(
            n_particles=n_particles,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )
    elif model_name == 'cgenn':
        return CGENNNBodyNet(
            n_particles=n_particles,
            hidden_channels=hidden_dim // 4,
            n_layers=n_layers,
        )
    elif model_name == 'nequip':
        return NequIPNBodyNet(
            n_particles=n_particles,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )
    elif model_name == 'baseline':
        return BaselineNBodyNetWithAttention(
            n_particles=n_particles,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_forward_time(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_warmup: int = 10,
    n_iterations: int = 50,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Measure forward pass time with proper warmup and synchronization.

    Returns:
        Dict with 'mean_ms', 'std_ms', 'min_ms', 'max_ms'
    """
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_tensor)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Timed runs
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
            times.append((t1 - t0) * 1000)  # ms

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
    }


def measure_backward_time(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_warmup: int = 5,
    n_iterations: int = 20,
    device: str = 'cuda'
) -> Dict[str, float]:
    """Measure forward + backward pass time."""
    model.train()

    # Warmup
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


def measure_peak_memory(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: str = 'cuda'
) -> Optional[float]:
    """Measure peak GPU memory in MB. Returns None for CPU."""
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
    """
    Fit O(N^alpha) to timing data via log-log regression.

    Returns:
        Dict with 'alpha' (exponent), 'r_squared' (fit quality)
    """
    ns = sorted(timings.keys())
    if len(ns) < 3:
        return {'alpha': float('nan'), 'r_squared': float('nan')}

    log_n = np.log(np.array(ns, dtype=float))
    log_t = np.log(np.array([timings[n] for n in ns], dtype=float))

    # Linear regression: log(t) = alpha * log(n) + c
    A = np.vstack([log_n, np.ones_like(log_n)]).T
    result = np.linalg.lstsq(A, log_t, rcond=None)
    alpha, c = result[0]

    # R^2
    ss_res = np.sum((log_t - (alpha * log_n + c)) ** 2)
    ss_tot = np.sum((log_t - np.mean(log_t)) ** 2)
    r_squared = 1 - ss_res / (ss_tot + 1e-12)

    return {'alpha': float(alpha), 'r_squared': float(r_squared)}


def run_scaling_experiment(config: ScalingConfig) -> Dict:
    """Run the full scaling experiment."""
    os.makedirs(config.save_dir, exist_ok=True)

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
                # Create model and input
                model = create_model(model_name, n_particles, config).to(device)
                input_tensor = torch.randn(
                    config.batch_size, n_particles, 4, device=device
                )

                n_params = count_parameters(model)
                results[model_name]['n_params'][n_particles] = n_params

                # Measure forward time
                fwd = measure_forward_time(
                    model, input_tensor,
                    n_warmup=config.n_warmup,
                    n_iterations=config.n_iterations,
                    device=device
                )
                results[model_name]['forward_times'][n_particles] = fwd['mean_ms']

                # Measure backward time
                bwd = measure_backward_time(
                    model, input_tensor,
                    n_warmup=config.n_warmup // 2,
                    n_iterations=config.n_iterations // 2,
                    device=device
                )
                results[model_name]['backward_times'][n_particles] = bwd['mean_ms']

                # Measure memory
                mem = measure_peak_memory(model, input_tensor, device)
                if mem is not None:
                    results[model_name]['peak_memory_mb'][n_particles] = mem

                print(f"fwd={fwd['mean_ms']:>8.2f}ms  "
                      f"bwd={bwd['mean_ms']:>8.2f}ms  "
                      f"params={n_params:>10,}", end='')
                if mem is not None:
                    print(f"  mem={mem:>8.1f}MB", end='')
                print()

                # Cleanup
                del model, input_tensor
                if device == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                results[model_name]['errors'][n_particles] = str(e)
                print(f"ERROR: {e}")

        # Compute complexity class from forward times
        fwd_times = results[model_name]['forward_times']
        if len(fwd_times) >= 3:
            complexity = compute_complexity_class(fwd_times)
            results[model_name]['complexity'] = complexity
            print(f"  Fitted complexity: O(N^{complexity['alpha']:.2f}) "
                  f"(R^2={complexity['r_squared']:.3f})")

    # Summary table
    print_scaling_summary(results, config)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(config.save_dir, f"scaling_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump({
            'config': asdict(config),
            'results': results,
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")
    return results


def print_scaling_summary(results: Dict, config: ScalingConfig):
    """Print formatted scaling summary."""
    print("\n" + "=" * 70)
    print("SCALING SUMMARY")
    print("=" * 70)

    # Forward time table
    print("\nForward Pass Time (ms) by N:")
    header = f"{'Model':<14}"
    for n in config.n_particles_list:
        header += f"  {'N='+str(n):>10}"
    header += f"  {'O(N^α)':>10}"
    print(header)
    print("-" * len(header))

    for model_name in config.models:
        if model_name not in results:
            continue
        row = f"{model_name:<14}"
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

    # Speedup table (HC-Net vs others at largest N)
    if config.n_particles_list:
        max_n = max(config.n_particles_list)
        hcnet_time = results.get('hcnet', {}).get('forward_times', {}).get(max_n)
        if hcnet_time and hcnet_time > 0:
            print(f"\nSpeedup vs HC-Net at N={max_n}:")
            for model_name in config.models:
                if model_name == 'hcnet':
                    continue
                other_time = results.get(model_name, {}).get(
                    'forward_times', {}
                ).get(max_n)
                if other_time:
                    ratio = other_time / hcnet_time
                    print(f"  {model_name:<14}: {ratio:.1f}x slower")

    # Memory table
    has_memory = any(
        results[m].get('peak_memory_mb')
        for m in config.models if m in results
    )
    if has_memory:
        print(f"\nPeak Memory (MB) by N:")
        header = f"{'Model':<14}"
        for n in config.n_particles_list:
            header += f"  {'N='+str(n):>10}"
        print(header)
        print("-" * len(header))

        for model_name in config.models:
            if model_name not in results:
                continue
            row = f"{model_name:<14}"
            mem = results[model_name].get('peak_memory_mb', {})
            for n in config.n_particles_list:
                if n in mem:
                    row += f"  {mem[n]:>10.1f}"
                else:
                    row += f"  {'N/A':>10}"
            print(row)

    # Complexity classification
    print(f"\nComplexity Classification:")
    print(f"{'Model':<14}  {'Exponent':>10}  {'R²':>8}  {'Class':>12}")
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
            print(f"{model_name:<14}  {alpha:>10.2f}  {r2:>8.3f}  {cls:>12}")
        else:
            print(f"{model_name:<14}  {'N/A':>10}  {'N/A':>8}  {'N/A':>12}")


def main():
    parser = argparse.ArgumentParser(description='Scaling Experiment')

    parser.add_argument('--models', nargs='+',
                        default=['hcnet', 'hcnet_meanfield', 'hcnet_attn',
                                 'egnn', 'cgenn', 'nequip', 'baseline'],
                        help='Models to compare')
    parser.add_argument('--n-particles', nargs='+', type=int,
                        default=[5, 10, 20, 50, 100],
                        help='Number of particles to test')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--n-iterations', type=int, default=50,
                        help='Timing iterations')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--save-dir', type=str,
                        default='./results/scaling',
                        help='Save directory')

    args = parser.parse_args()

    config = ScalingConfig(
        models=args.models,
        n_particles_list=args.n_particles,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        n_iterations=args.n_iterations,
        device=args.device,
        save_dir=args.save_dir,
    )

    run_scaling_experiment(config)


if __name__ == '__main__':
    main()
