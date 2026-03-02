#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH -n 1
#SBATCH --time=00:30:00
#SBATCH --output=/home/ltan/HHC/scripts/moead_comparison_job_%j.out

# MOEA/D vs DRL comparison plot (no GPU needed).
# Usage: sbatch scripts/moead_comparison_job.sh [raw_200ep|raw_101w|exp1]

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hhc
cd /home/ltan/HHC

DRL_EXP="${1:-raw_200ep}"
echo "=== MOEA/D vs DRL comparison (DRL exp: $DRL_EXP) ==="

bash scripts/run_moead_comparison_plot.sh "$DRL_EXP"
