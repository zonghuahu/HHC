#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/home/ltan/HHC/scripts/pareto_inference_job_%j.out

# DRL Pareto inference for 200-epoch trained models (n=50, n=100).
# Output: paretofront/pareto_results_{50,100}_raw_200ep.pkl
# Run MOEA/D comparison after: bash scripts/run_moead_comparison_plot.sh raw_200ep

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hhc
cd /home/ltan/HHC

echo "=== DRL Pareto Inference (200ep models) on shared test data ==="

# n=50: train_raw_50_200ep
LOAD_50="outputs/agh_50/train_raw_50_200ep_20260228T162007"
if [ -f "$LOAD_50/epoch-199.pt" ]; then
    echo "[n=50] Using $LOAD_50/epoch-199.pt"
    python -u visualization/pareto_inference.py \
        --load_path "$LOAD_50/epoch-199.pt" \
        --graph_size 50 \
        --val_size 1000 \
        --val_dataset paretofront/shared_test_50.pkl \
        --seed 1234 \
        --exp_name raw_200ep \
        --output_dir paretofront
else
    echo "[WARN] $LOAD_50/epoch-199.pt not found, skipping n=50"
fi

# n=100: train_raw_100_200ep
LOAD_100="outputs/agh_100/train_raw_100_200ep_20260228T162008"
if [ -f "$LOAD_100/epoch-199.pt" ]; then
    echo "[n=100] Using $LOAD_100/epoch-199.pt"
    python -u visualization/pareto_inference.py \
        --load_path "$LOAD_100/epoch-199.pt" \
        --graph_size 100 \
        --val_size 1000 \
        --val_dataset paretofront/shared_test_100.pkl \
        --seed 1234 \
        --exp_name raw_200ep \
        --output_dir paretofront
else
    echo "[WARN] $LOAD_100/epoch-199.pt not found, skipping n=100"
fi

echo "=== Done. Results: paretofront/pareto_results_{50,100}_raw_200ep.pkl ==="
