#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/home/ltan/HHC/scripts/moead/moead_50_%j.out

# MOEA/D baseline for HHCRSP, n=50, using shared test instances
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hhc
cd /home/ltan/HHC

python -u moead.py \
    --graph_size 50 \
    --n_instances 1000 \
    --filename paretofront/shared_test_50.pkl \
    --n_weights 11 \
    --n_gen 500 \
    --T 3 \
    --mutation_rate 0.3 \
    --seed 1234 \
    --output_dir scripts/moead
