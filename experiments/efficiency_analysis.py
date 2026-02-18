"""
Computational Efficiency Analysis.

Measures and compares computational costs across models:
- Parameter counts
- FLOPs estimation
- Memory usage
- Inference time
- Training throughput

Addresses reviewer concern: "Missing computational efficiency analysis"
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
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


# Import 2D models
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

# Import 3D models
_nbody_models_3d = _import_module(
    "nbody_models_3d", os.path.join(_PCNN_ROOT, 'models', 'nbody_models_3d.py')
)
CliffordNBodyNet3D = _nbody_models_3d.CliffordNBodyNet3D
BaselineNBodyNetWithAttention3D = _nbody_models_3d.BaselineNBodyNetWithAttention3D


@dataclass
class EfficiencyConfig:
    """Configuration for efficiency analysis."""
    models_2d: List[str] = None
    models_3d: List[str] = None
    n_particles_list: List[int] = None
    batch_sizes: List[int] = None
    hidden_dims: List[int] = None
    n_layers: int = 4
    n_warmup: int = 10
    n_iterations: int = 100
    device: str = 'cuda'
    save_dir: str = './results/efficiency'

    def __post_init__(self):
        if self.models_2d is None:
            self.models_2d = ['hcnet', 'egnn', 'cgenn', 'nequip', 'baseline']
        if self.models_3d is None:
            self.models_3d = ['hcnet3d', 'egnn3d', 'baseline3d']
        if self.n_particles_list is None:
            self.n_particles_list = [5, 10, 20, 50]
        if self.batch_sizes is None:
            self.batch_sizes = [1, 8, 32, 128]
        if self.hidden_dims is None:
            self.hidden_dims = [64, 128, 256]


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_by_layer(model: nn.Module) -> Dict[str, int]:
    """Count parameters per named module."""
    counts = {}
    for name, module in model.named_modules():
        if name:
            n_params = sum(p.numel() for p in module.parameters(recurse=False))
            if n_params > 0:
                counts[name] = n_params
    return counts


def estimate_flops_linear(in_features: int, out_features: int, batch_size: int = 1) -> int:
    """Estimate FLOPs for a linear layer (multiply-add)."""
    return 2 * batch_size * in_features * out_features


def estimate_flops_attention(seq_len: int, hidden_dim: int, n_heads: int, batch_size: int = 1) -> int:
    """Estimate FLOPs for multi-head attention."""
    # Q, K, V projections
    qkv_flops = 3 * estimate_flops_linear(hidden_dim, hidden_dim, batch_size * seq_len)
    # Attention scores: Q @ K^T
    score_flops = batch_size * n_heads * seq_len * seq_len * (hidden_dim // n_heads)
    # Softmax (approximate)
    softmax_flops = batch_size * n_heads * seq_len * seq_len * 5
    # Attention @ V
    attn_flops = batch_size * n_heads * seq_len * seq_len * (hidden_dim // n_heads)
    # Output projection
    out_flops = estimate_flops_linear(hidden_dim, hidden_dim, batch_size * seq_len)

    return qkv_flops + score_flops + softmax_flops + attn_flops + out_flops


def estimate_model_flops(
    model_name: str,
    n_particles: int,
    hidden_dim: int,
    n_layers: int,
    batch_size: int,
    coord_dim: int = 2
) -> int:
    """
    Estimate FLOPs for a model.

    This is a rough estimation based on model architecture.
    """
    input_dim = coord_dim * 2  # pos + vel

    # Input projection
    flops = estimate_flops_linear(input_dim, hidden_dim, batch_size * n_particles)

    # Processing layers
    for _ in range(n_layers):
        # MLP (fc1 + fc2)
        flops += estimate_flops_linear(hidden_dim, hidden_dim * 2, batch_size * n_particles)
        flops += estimate_flops_linear(hidden_dim * 2, hidden_dim, batch_size * n_particles)

        # Model-specific operations
        if model_name in ['hcnet', 'hcnet3d']:
            # Geometric mixing: outer product + projection
            group_size = 8 if model_name == 'hcnet3d' else 4
            n_groups = hidden_dim // group_size
            # Outer product
            flops += batch_size * n_particles * n_groups * group_size * group_size
            # Projection
            flops += batch_size * n_particles * n_groups * group_size * group_size * group_size

        elif model_name in ['egnn', 'egnn3d']:
            # Pairwise distance computation
            flops += batch_size * n_particles * n_particles * coord_dim * 3
            # Edge MLP
            edge_input_dim = hidden_dim * 2 + 1
            flops += estimate_flops_linear(edge_input_dim, hidden_dim, batch_size * n_particles * n_particles)

        elif model_name in ['cgenn', 'cgenn3d']:
            # Multivector operations (simplified)
            mv_dim = 16  # Cl(4,0)
            flops += batch_size * n_particles * hidden_dim * mv_dim * 2

        elif model_name in ['nequip', 'nequip3d']:
            # Radial basis + angular encoding
            n_radial = 8
            n_angular = 9
            flops += batch_size * n_particles * n_particles * (n_radial + n_angular)
            # Edge network
            edge_input_dim = hidden_dim * 2 + n_radial + n_angular
            flops += estimate_flops_linear(edge_input_dim, hidden_dim, batch_size * n_particles * n_particles)

    # Attention (if applicable)
    if model_name not in ['cgenn', 'cgenn3d']:
        flops += estimate_flops_attention(n_particles, hidden_dim, 4, batch_size)

    # Output projection
    flops += estimate_flops_linear(hidden_dim, input_dim, batch_size * n_particles)

    return flops


def measure_memory(model: nn.Module, input_tensor: torch.Tensor, device: str) -> Dict[str, float]:
    """
    Measure GPU memory usage during forward and backward pass.

    Returns memory in MB.
    """
    if device == 'cpu':
        return {'forward_mb': 0, 'backward_mb': 0, 'peak_mb': 0}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # Forward pass
    torch.cuda.synchronize()
    output = model(input_tensor)
    torch.cuda.synchronize()
    forward_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Backward pass
    loss = output.sum()
    loss.backward()
    torch.cuda.synchronize()
    backward_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Cleanup
    del output, loss
    torch.cuda.empty_cache()

    return {
        'forward_mb': forward_mem,
        'backward_mb': backward_mem,
        'peak_mb': peak_mem
    }


def measure_inference_time(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: str,
    n_warmup: int = 10,
    n_iterations: int = 100
) -> Dict[str, float]:
    """
    Measure inference time.

    Returns time in milliseconds.
    """
    model = model.to(device)
    model.eval()
    input_tensor = input_tensor.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_tensor)

    if device != 'cpu':
        torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(n_iterations):
            if device != 'cpu':
                torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(input_tensor)

            if device != 'cpu':
                torch.cuda.synchronize()
            end = time.perf_counter()

            times.append((end - start) * 1000)  # Convert to ms

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times)
    }


def measure_training_throughput(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    device: str,
    n_iterations: int = 50
) -> Dict[str, float]:
    """
    Measure training throughput (samples/second).
    """
    model = model.to(device)
    model.train()
    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = nn.functional.mse_loss(output, target_tensor)
        loss.backward()
        optimizer.step()

    if device != 'cpu':
        torch.cuda.synchronize()

    # Measure
    batch_size = input_tensor.shape[0]
    start = time.perf_counter()

    for _ in range(n_iterations):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = nn.functional.mse_loss(output, target_tensor)
        loss.backward()
        optimizer.step()

    if device != 'cpu':
        torch.cuda.synchronize()
    end = time.perf_counter()

    total_samples = batch_size * n_iterations
    total_time = end - start

    return {
        'samples_per_second': total_samples / total_time,
        'iterations_per_second': n_iterations / total_time,
        'ms_per_iteration': (total_time / n_iterations) * 1000
    }


def create_model(model_name: str, n_particles: int, hidden_dim: int, n_layers: int) -> nn.Module:
    """Create model by name."""
    if model_name == 'hcnet':
        return CliffordNBodyNet(n_particles=n_particles, hidden_dim=hidden_dim, n_layers=n_layers)
    elif model_name == 'egnn':
        return EGNNNBodyNet(n_particles=n_particles, hidden_dim=hidden_dim, n_layers=n_layers, coord_dim=2)
    elif model_name == 'cgenn':
        return CGENNNBodyNet(n_particles=n_particles, hidden_channels=hidden_dim//4, n_layers=n_layers, coord_dim=2)
    elif model_name == 'nequip':
        return NequIPNBodyNet(n_particles=n_particles, hidden_dim=hidden_dim, n_layers=n_layers, coord_dim=2)
    elif model_name == 'baseline':
        return BaselineNBodyNetWithAttention(n_particles=n_particles, hidden_dim=hidden_dim, n_layers=n_layers)
    elif model_name == 'hcnet3d':
        return CliffordNBodyNet3D(n_particles=n_particles, hidden_dim=hidden_dim, n_layers=n_layers)
    elif model_name == 'egnn3d':
        return EGNNNBodyNet(n_particles=n_particles, hidden_dim=hidden_dim, n_layers=n_layers, coord_dim=3)
    elif model_name == 'baseline3d':
        return BaselineNBodyNetWithAttention3D(n_particles=n_particles, hidden_dim=hidden_dim, n_layers=n_layers)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_efficiency_analysis(config: EfficiencyConfig) -> Dict:
    """Run complete efficiency analysis."""
    os.makedirs(config.save_dir, exist_ok=True)

    results = {
        'config': asdict(config),
        'parameter_counts': {},
        'flops_estimates': {},
        'memory_usage': {},
        'inference_times': {},
        'training_throughput': {}
    }

    device = config.device if torch.cuda.is_available() else 'cpu'
    print(f"Running efficiency analysis on {device}")

    # Analyze 2D models
    print("\n=== 2D Models ===")
    for model_name in config.models_2d:
        print(f"\nAnalyzing {model_name}...")
        results['parameter_counts'][model_name] = {}
        results['flops_estimates'][model_name] = {}
        results['memory_usage'][model_name] = {}
        results['inference_times'][model_name] = {}
        results['training_throughput'][model_name] = {}

        for hidden_dim in config.hidden_dims:
            for n_particles in config.n_particles_list:
                key = f"h{hidden_dim}_n{n_particles}"

                try:
                    model = create_model(model_name, n_particles, hidden_dim, config.n_layers)

                    # Parameter count
                    n_params = count_parameters(model)
                    results['parameter_counts'][model_name][key] = n_params

                    # FLOPs estimate
                    flops = estimate_model_flops(
                        model_name, n_particles, hidden_dim,
                        config.n_layers, batch_size=32, coord_dim=2
                    )
                    results['flops_estimates'][model_name][key] = flops

                    # Memory and timing for default batch size
                    input_tensor = torch.randn(32, n_particles, 4)
                    target_tensor = torch.randn(32, n_particles, 4)

                    # Memory
                    if device != 'cpu':
                        mem = measure_memory(model, input_tensor, device)
                        results['memory_usage'][model_name][key] = mem

                    # Inference time
                    timing = measure_inference_time(
                        model, input_tensor, device,
                        config.n_warmup, config.n_iterations
                    )
                    results['inference_times'][model_name][key] = timing

                    # Training throughput
                    throughput = measure_training_throughput(
                        model, input_tensor, target_tensor, device
                    )
                    results['training_throughput'][model_name][key] = throughput

                    print(f"  {key}: {n_params:,} params, {timing['mean_ms']:.2f}ms inference")

                except Exception as e:
                    print(f"  {key}: ERROR - {e}")

    # Analyze 3D models
    print("\n=== 3D Models ===")
    for model_name in config.models_3d:
        print(f"\nAnalyzing {model_name}...")
        results['parameter_counts'][model_name] = {}
        results['flops_estimates'][model_name] = {}
        results['inference_times'][model_name] = {}
        results['training_throughput'][model_name] = {}

        for hidden_dim in config.hidden_dims:
            for n_particles in config.n_particles_list:
                key = f"h{hidden_dim}_n{n_particles}"

                try:
                    model = create_model(model_name, n_particles, hidden_dim, config.n_layers)

                    # Parameter count
                    n_params = count_parameters(model)
                    results['parameter_counts'][model_name][key] = n_params

                    # FLOPs estimate
                    flops = estimate_model_flops(
                        model_name, n_particles, hidden_dim,
                        config.n_layers, batch_size=32, coord_dim=3
                    )
                    results['flops_estimates'][model_name][key] = flops

                    # Timing
                    input_tensor = torch.randn(32, n_particles, 6)
                    target_tensor = torch.randn(32, n_particles, 6)

                    timing = measure_inference_time(
                        model, input_tensor, device,
                        config.n_warmup, config.n_iterations
                    )
                    results['inference_times'][model_name][key] = timing

                    throughput = measure_training_throughput(
                        model, input_tensor, target_tensor, device
                    )
                    results['training_throughput'][model_name][key] = throughput

                    print(f"  {key}: {n_params:,} params, {timing['mean_ms']:.2f}ms inference")

                except Exception as e:
                    print(f"  {key}: ERROR - {e}")

    # Scaling analysis
    print("\n=== Scaling Analysis ===")
    results['scaling'] = analyze_scaling(config, device)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(config.save_dir, f"efficiency_{timestamp}.json")

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    # Print summary
    print_efficiency_summary(results)

    return results


def analyze_scaling(config: EfficiencyConfig, device: str) -> Dict:
    """Analyze how models scale with input size."""
    scaling = {}

    hidden_dim = 128
    n_layers = 4
    batch_size = 32

    for model_name in ['hcnet', 'egnn', 'baseline']:
        scaling[model_name] = {'n_particles': [], 'inference_ms': [], 'params': []}

        for n_particles in [5, 10, 20, 50, 100]:
            try:
                model = create_model(model_name, n_particles, hidden_dim, n_layers)
                input_tensor = torch.randn(batch_size, n_particles, 4)

                timing = measure_inference_time(model, input_tensor, device, n_warmup=5, n_iterations=20)

                scaling[model_name]['n_particles'].append(n_particles)
                scaling[model_name]['inference_ms'].append(timing['mean_ms'])
                scaling[model_name]['params'].append(count_parameters(model))

            except Exception as e:
                print(f"Scaling {model_name} n={n_particles}: {e}")

    return scaling


def print_efficiency_summary(results: Dict):
    """Print efficiency analysis summary."""
    print("\n" + "=" * 80)
    print("EFFICIENCY ANALYSIS SUMMARY")
    print("=" * 80)

    # Parameter comparison
    print("\n--- Parameter Counts (hidden_dim=128, n_particles=5) ---")
    key = "h128_n5"
    for model_name in results['parameter_counts']:
        if key in results['parameter_counts'][model_name]:
            params = results['parameter_counts'][model_name][key]
            print(f"  {model_name:<12}: {params:>10,} parameters")

    # Inference time comparison
    print("\n--- Inference Time (hidden_dim=128, n_particles=5, batch=32) ---")
    for model_name in results['inference_times']:
        if key in results['inference_times'][model_name]:
            timing = results['inference_times'][model_name][key]
            print(f"  {model_name:<12}: {timing['mean_ms']:>8.2f} ± {timing['std_ms']:.2f} ms")

    # Training throughput
    print("\n--- Training Throughput (samples/second) ---")
    for model_name in results['training_throughput']:
        if key in results['training_throughput'][model_name]:
            tp = results['training_throughput'][model_name][key]
            print(f"  {model_name:<12}: {tp['samples_per_second']:>8.1f} samples/s")

    # Scaling summary
    if 'scaling' in results:
        print("\n--- Scaling with N particles (inference time) ---")
        print(f"  {'Model':<12} {'N=5':>10} {'N=20':>10} {'N=50':>10} {'N=100':>10}")
        print("-" * 55)
        for model_name, data in results['scaling'].items():
            row = f"  {model_name:<12}"
            for n in [5, 20, 50, 100]:
                if n in data['n_particles']:
                    idx = data['n_particles'].index(n)
                    row += f" {data['inference_ms'][idx]:>9.2f}ms"
                else:
                    row += f" {'N/A':>10}"
            print(row)


def main():
    parser = argparse.ArgumentParser(description='Efficiency Analysis')

    parser.add_argument('--models-2d', nargs='+',
                       default=['hcnet', 'egnn', 'cgenn', 'nequip', 'baseline'],
                       help='2D models to analyze')
    parser.add_argument('--models-3d', nargs='+',
                       default=['hcnet3d', 'egnn3d', 'baseline3d'],
                       help='3D models to analyze')
    parser.add_argument('--n-particles', nargs='+', type=int,
                       default=[5, 10, 20],
                       help='Number of particles')
    parser.add_argument('--hidden-dims', nargs='+', type=int,
                       default=[64, 128],
                       help='Hidden dimensions')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    parser.add_argument('--save-dir', type=str,
                       default='./results/efficiency',
                       help='Save directory')

    args = parser.parse_args()

    config = EfficiencyConfig(
        models_2d=args.models_2d,
        models_3d=args.models_3d,
        n_particles_list=args.n_particles,
        hidden_dims=args.hidden_dims,
        device=args.device,
        save_dir=args.save_dir
    )

    run_efficiency_analysis(config)


if __name__ == '__main__':
    main()
