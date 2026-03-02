#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=5-00:00:00
#SBATCH --output=/home/ltan/HHC/scripts/train_raw_50_200ep_%j.out

# DRL n=50, 200 epochs (convergence verification)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hhc
cd /home/ltan/HHC
python -u run.py --n_epochs 200 --graph_size 50 --problem agh --baseline rollout \
    --run_name train_raw_50_200ep
