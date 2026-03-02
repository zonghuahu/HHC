#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/home/ltan/HHC/scripts/plot_weadd_%j.out

# Visualization: convergence + Pareto front for WE-Add models
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hhc
cd /home/ltan/HHC

echo "=== Plotting convergence and Pareto fronts ==="
python -u visualization/plot_pareto_and_convergence.py --all --size 50 100 --exp_name weadd
echo "=== Done. Check images/ directory ==="
