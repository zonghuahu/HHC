#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/home/ltan/HHC/scripts/train_weadd_100_%j.out

# DRL WE-Add multi-objective training, n=100, 200 epochs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hhc
cd /home/ltan/HHC
python -u run.py --n_epochs 200 --graph_size 100 --problem agh --baseline rollout \
    --run_name train_weadd_100
