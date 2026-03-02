#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/home/ltan/HHC/scripts/pareto_weadd_%j.out

# DRL Pareto inference for WE-Add trained models (n=50, n=100).
# NOTE: Update LOAD_50 and LOAD_100 paths after training completes.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hhc
cd /home/ltan/HHC

echo "=== DRL Pareto Inference (WE-Add models) on shared test data ==="

# --- n=50 ---
# Find the latest train_weadd_50 run directory
LOAD_50=$(ls -td outputs/agh_50/train_weadd_50_* 2>/dev/null | head -1)
if [ -n "$LOAD_50" ]; then
    EPOCH_50=$(ls "$LOAD_50"/epoch-*.pt 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -n "$EPOCH_50" ]; then
        echo "[n=50] Using $EPOCH_50"
        python -u visualization/pareto_inference.py \
            --load_path "$EPOCH_50" \
            --graph_size 50 \
            --val_size 1000 \
            --val_dataset paretofront/shared_test_50.pkl \
            --seed 1234 \
            --exp_name weadd \
            --output_dir paretofront
    else
        echo "[WARN] No epoch checkpoint found in $LOAD_50"
    fi
else
    echo "[WARN] No train_weadd_50 directory found"
fi

# --- n=100 ---
LOAD_100=$(ls -td outputs/agh_100/train_weadd_100_* 2>/dev/null | head -1)
if [ -n "$LOAD_100" ]; then
    EPOCH_100=$(ls "$LOAD_100"/epoch-*.pt 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -n "$EPOCH_100" ]; then
        echo "[n=100] Using $EPOCH_100"
        python -u visualization/pareto_inference.py \
            --load_path "$EPOCH_100" \
            --graph_size 100 \
            --val_size 1000 \
            --val_dataset paretofront/shared_test_100.pkl \
            --seed 1234 \
            --exp_name weadd \
            --output_dir paretofront
    else
        echo "[WARN] No epoch checkpoint found in $LOAD_100"
    fi
else
    echo "[WARN] No train_weadd_100 directory found"
fi

echo "=== Done ==="
