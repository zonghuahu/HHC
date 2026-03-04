#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/home/ltan/HHC/scripts/plot_pareto_both_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hhc
cd /home/ltan/HHC

RUN_50="outputs/agh_50/train_weadd_50_20260303T002028"
RUN_100="outputs/agh_100/train_weadd_100_20260303T002036"

echo "=== Step 1: DRL Pareto Inference n=100 ==="
EPOCH_100=$(ls "$RUN_100"/epoch-*.pt 2>/dev/null | sort -t- -k2 -n | tail -1)
python -u visualization/pareto_inference.py \
    --load_path "$EPOCH_100" \
    --graph_size 100 \
    --val_size 1000 \
    --val_dataset paretofront/shared_test_100.pkl \
    --seed 1234 \
    --exp_name weadd101 \
    --output_dir paretofront

echo ""
echo "=== Step 2: Pareto Comparison n=50 ==="
python -u visualization/plot_moead_comparison.py \
    --graph_size 50 \
    --drl_path paretofront/pareto_results_50_tmp50.pkl \
    --moead_path paretofront/moead_results_50_exp1.pkl \
    --output_dir paretofront

echo ""
echo "=== Step 3: Pareto Comparison n=100 ==="
python -u visualization/plot_moead_comparison.py \
    --graph_size 100 \
    --drl_path paretofront/pareto_results_100_weadd101.pkl \
    --moead_path paretofront/moead_results_100_exp1.pkl \
    --output_dir paretofront

echo ""
echo "=== Done ==="
