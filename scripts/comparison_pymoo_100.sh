#!/bin/bash
#SBATCH -p gpu_a100
#SBATCH --gpus=1
#SBATCH -t 5-00:00:00
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -o /home/ltan/HHC/scripts/comparison_pymoo_100_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hhc
cd /home/ltan/HHC

echo "=== pymoo MOEA/D (n=100) on shared test data ==="
python -u hhc_problem_pymoo.py \
    --algorithm MOEAD \
    --graph_size 100 \
    --filename paretofront/shared_test_100.pkl \
    --n_instances 1000 \
    --pop_size 101 \
    --n_gen 500 \
    --seed 1234 \
    --output_dir paretofront \
    --moead_format
