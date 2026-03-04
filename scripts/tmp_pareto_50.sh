#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/home/ltan/HHC/scripts/tmp_pareto_50_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hhc
cd /home/ltan/HHC

RUN_50="outputs/agh_50/train_weadd_50_20260303T002028"
EPOCH=$(ls "$RUN_50"/epoch-*.pt 2>/dev/null | sort -t- -k2 -n | tail -1)

echo "=== Step 1: DRL Pareto Inference (n=50) ==="
python -u visualization/pareto_inference.py \
    --load_path "$EPOCH" \
    --graph_size 50 \
    --val_size 1000 \
    --val_dataset paretofront/shared_test_50.pkl \
    --seed 1234 \
    --exp_name tmp50 \
    --output_dir paretofront

echo ""
echo "=== Step 2: DRL vs MOEA/D Comparison ==="
if [ -f paretofront/moead_results_50_exp1.pkl ]; then
    python -u visualization/plot_moead_comparison.py \
        --graph_size 50 \
        --drl_path paretofront/pareto_results_50_tmp50.pkl \
        --moead_path paretofront/moead_results_50_exp1.pkl \
        --output_dir paretofront
else
    echo "[WARN] paretofront/moead_results_50_exp1.pkl not found. Run MOEA/D first:"
    echo "  python -u moead.py --graph_size 50 --filename paretofront/shared_test_50.pkl --output_dir paretofront"
fi

echo ""
echo "=== Done. Outputs: paretofront/pareto_results_50_tmp50.pkl, paretofront/moead_vs_drl_50.png ==="
