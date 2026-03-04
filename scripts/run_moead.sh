#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/home/ltan/HHC/scripts/run_moead_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hhc
cd /home/ltan/HHC

echo "=== MOEA/D n=50 (shared_test_50.pkl, 1000 instances) ==="
python -u moead.py \
    --graph_size 50 \
    --filename paretofront/shared_test_50.pkl \
    --n_instances 1000 \
    --n_weights 101 \
    --n_gen 500 \
    --seed 1234 \
    --output_dir paretofront

echo ""
echo "=== MOEA/D n=100 (shared_test_100.pkl, 1000 instances) ==="
python -u moead.py \
    --graph_size 100 \
    --filename paretofront/shared_test_100.pkl \
    --n_instances 1000 \
    --n_weights 101 \
    --n_gen 500 \
    --seed 1234 \
    --output_dir paretofront

echo ""
echo "=== Done. Outputs: paretofront/moead_results_{50,100}_exp1.pkl ==="
