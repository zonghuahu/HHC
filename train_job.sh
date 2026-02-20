#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=/home/ltan/HHC/train_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hhc
cd /home/ltan/HHC
python run.py --n_epochs 100
