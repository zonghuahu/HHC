#!/bin/bash
#SBATCH -p gpu_a100
#SBATCH --gpus=1
#SBATCH -t 5-00:00:00
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -o /home/ltan/HHC/scripts/comparison_pymoo_50_%j.out

# pymoo MOEA/D on shared test data — same conditions as moead.py for fair comparison
# Output: paretofront/pymoo_moead_results_50_exp1.pkl (moead format for plot_moead_comparison)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hhc
cd /home/ltan/HHC

echo "=== pymoo MOEA/D (n=50) on shared test data ==="
python -u hhc_problem_pymoo.py \
    --algorithm MOEAD \
    --graph_size 50 \
    --filename paretofront/shared_test_50.pkl \
    --n_instances 1000 \
    --pop_size 101 \
    --n_gen 500 \
    --seed 1234 \
    --output_dir paretofront \
    --moead_format

echo ""
echo "=== Plot DRL vs pymoo MOEA/D ==="
DRL_PKL="paretofront/pareto_results_50_raw_101w.pkl"
[ ! -f "$DRL_PKL" ] && DRL_PKL="paretofront/pareto_results_50_exp1.pkl"
python -u visualization/plot_moead_comparison.py \
    --graph_size 50 \
    --drl_path "$DRL_PKL" \
    --moead_path paretofront/pymoo_moead_results_50_exp1.pkl \
    --exp_name pymoo_comparison
