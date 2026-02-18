"""
Unified training entry point for HC-Net experiments.

Usage:
    python -m nips_hcnet.train --experiment exp1 --device cuda
    python -m nips_hcnet.train --experiment exp2 --mode spiral --epochs 100
    python -m nips_hcnet.train --experiment exp4 --molecule ethanol
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='HC-Net Training')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['exp1', 'exp2', 'exp3', 'exp4', 'exp5', 'exp6', 'exp7'],
                        help='Experiment to run')
    args, remaining = parser.parse_known_args()

    # Dispatch to the appropriate experiment module
    if args.experiment == 'exp1':
        from nips_hcnet.experiments.exp1_spinning_system_2d import main as run
    elif args.experiment == 'exp2':
        from nips_hcnet.experiments.exp2_chirality_grade_hierarchy import main as run
    elif args.experiment == 'exp3':
        from nips_hcnet.experiments.exp3_3d_nbody_chirality import main as run
    elif args.experiment == 'exp4':
        from nips_hcnet.experiments.exp4_md17_hybrid import main as run
    elif args.experiment == 'exp5':
        from nips_hcnet.experiments.exp5_scaling_analysis import main as run
    elif args.experiment == 'exp6':
        from nips_hcnet.experiments.exp6_sota_comparison_3d import main as run
    elif args.experiment == 'exp7':
        from nips_hcnet.experiments.exp7_ablation import main as run

    # Reset sys.argv so the experiment's argparse works correctly
    sys.argv = [sys.argv[0]] + remaining
    run()


if __name__ == '__main__':
    main()
