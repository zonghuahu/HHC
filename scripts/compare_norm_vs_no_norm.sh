#!/bin/bash
# Run Pareto inference and compare DRL vs MOEA/D (raw cost, 101 gates)
# Run AFTER training completes. Uses train_raw_50 / train_raw_100.
#
# Example after training:
#   outputs/agh_50/train_raw_50_YYYYMMDDTHHMMSS/epoch-99.pt
#   outputs/agh_100/train_raw_100_YYYYMMDDTHHMMSS/epoch-99.pt

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hhc
cd /home/ltan/HHC

# Auto-detect latest train_raw checkpoints
RAW_50=$(ls -td outputs/agh_50/train_raw_50_* 2>/dev/null | head -1)
RAW_100=$(ls -td outputs/agh_100/train_raw_100_* 2>/dev/null | head -1)

# Fallback: also check train_no_norm (legacy naming)
[ -z "$RAW_50" ] && RAW_50=$(ls -td outputs/agh_50/train_no_norm_50_* 2>/dev/null | head -1)
[ -z "$RAW_100" ] && RAW_100=$(ls -td outputs/agh_100/train_no_norm_100_* 2>/dev/null | head -1)

if [ -z "$RAW_50" ] || [ ! -f "$RAW_50/epoch-99.pt" ]; then
    echo "[WARN] train_raw_50 not found or incomplete. Run: sbatch scripts/train_no_norm_50.sh"
    RAW_50=""
fi
if [ -z "$RAW_100" ] || [ ! -f "$RAW_100/epoch-99.pt" ]; then
    echo "[WARN] train_raw_100 not found or incomplete. Run: sbatch scripts/train_no_norm_100.sh"
    RAW_100=""
fi

echo "=== DRL Pareto Inference: Raw cost n=50 ==="
if [ -n "$RAW_50" ]; then
    python -u visualization/pareto_inference.py \
        --load_path "$RAW_50/epoch-99.pt" \
        --graph_size 50 --val_size 1000 \
        --val_dataset paretofront/shared_test_50.pkl \
        --seed 1234 --exp_name raw_101w --output_dir paretofront
fi

echo "=== DRL Pareto Inference: Raw cost n=100 ==="
if [ -n "$RAW_100" ]; then
    python -u visualization/pareto_inference.py \
        --load_path "$RAW_100/epoch-99.pt" \
        --graph_size 100 --val_size 1000 \
        --val_dataset paretofront/shared_test_100.pkl \
        --seed 1234 --exp_name raw_101w_100 --output_dir paretofront
fi

if [ -f paretofront/pareto_results_50_raw_101w.pkl ]; then
    echo "=== MOEA/D vs DRL Comparison (n=50, raw cost) ==="
    python -u visualization/plot_moead_comparison.py \
        --graph_size 50 \
        --drl_path paretofront/pareto_results_50_raw_101w.pkl \
        --moead_path paretofront/moead_results_50_exp1.pkl \
        --exp_name raw_101w
else
    echo "[SKIP] pareto_results_50_raw_101w.pkl not found. Run training first."
fi
