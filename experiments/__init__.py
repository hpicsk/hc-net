"""
HC-Net Experiments.

NeurIPS paper experiments:
  Exp 1: Spinning system 2D — vector collapse demonstration
  Exp 2: Chirality grade hierarchy — trivector necessity proof
  Exp 3: 3D N-body chirality — physical system validation
  Exp 4: MD17 hybrid — molecular force prediction benchmark
  Exp 5: Scaling analysis — O(N) vs O(N^2)
  Exp 6: SOTA comparison 3D — baseline comparison
  Exp 7: Ablation study — component contribution analysis

Legacy experiments (from original HC-Net):
  ablation_study, angular_momentum_classification, efficiency_analysis,
  generalization, geometric_mnist_exp, md17_exp, nbody_physics_exp,
  meanfield_only_classification, sample_efficiency, relational_mnist_exp,
  scaling_exp, spinning_system_exp, sota_comparison_exp, sota_comparison_3d_exp,
  starved_nbody_exp, theoretical_analysis
"""

__all__ = []

try:
    from .sample_efficiency import run_sample_efficiency_experiment
    __all__.append('run_sample_efficiency_experiment')
except ImportError:
    pass

try:
    from .generalization import run_compositional_experiment
    __all__.append('run_compositional_experiment')
except ImportError:
    pass
