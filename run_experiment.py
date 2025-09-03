#!/usr/bin/env python3
"""
Launcher script for emitter classification experiments.
"""

import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Run emitter classification experiments')
    parser.add_argument('experiment', choices=['triplet', 'dual_encoder', 'supcon', 'ft_transformer'],
                       help='Type of experiment to run')
    parser.add_argument('--num_gpus', type=int, default=1,
                       help='Number of GPUs to use (default: 1)')
    parser.add_argument('--config', type=str, default=None,
                       help='Custom configuration file (optional)')
    
    args = parser.parse_args()
    
    # Map experiment names to script files
    experiment_scripts = {
        'triplet': 'train_triplet.py',
        'dual_encoder': 'train_dual_encoder.py', 
        'supcon': 'train_supcon.py',
        'ft_transformer': 'train_ft_transformer.py'
    }
    
    script = experiment_scripts[args.experiment]
    
    # Check if script exists
    if not os.path.exists(script):
        print(f"Error: Script {script} not found!")
        sys.exit(1)
    
    # Build command using torchrun (new PyTorch distributed launcher)
    cmd = [
        'torchrun',
        '--nproc_per_node', str(args.num_gpus),
        script
    ]
    
    # Add custom config if provided
    if args.config:
        cmd.extend(['--config', args.config])
    
    print(f"Running experiment: {args.experiment}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    # Run the experiment
    try:
        subprocess.run(cmd, check=True)
        print("-" * 50)
        print(f"Experiment {args.experiment} completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main() 