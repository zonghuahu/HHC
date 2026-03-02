#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=5-00:00:00
#SBATCH --output=/home/ltan/HHC/scripts/train_no_norm_50_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hhc
cd /home/ltan/HHC
python -u run.py --n_epochs 100 --graph_size 50 --problem agh --baseline rollout \
    --run_name train_raw_50
