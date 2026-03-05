#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/home/ltan/HHC/scripts/train_tche_100_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hhc
cd /home/ltan/HHC

echo "=== DRL Training n=100, 200 epochs (Tchebycheff scalarization) ==="
python -u run.py \
    --problem agh \
    --graph_size 100 \
    --n_epochs 200 \
    --run_name train_tche_100 \
    --val_dataset paretofront/shared_test_100.pkl \
    --val_size 1000 \
    --seed 1234

echo ""
echo "=== Done. Checkpoints: outputs/agh_100/train_tche_100_*/epoch-*.pt ==="
