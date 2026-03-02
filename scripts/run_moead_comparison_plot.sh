#!/bin/bash
# MOEA/D vs DRL Pareto front comparison.
# Usage:
#   bash scripts/run_moead_comparison_plot.sh              # default: exp1 DRL vs MOEA/D
#   bash scripts/run_moead_comparison_plot.sh raw_200ep    # 200ep DRL vs MOEA/D
#   bash scripts/run_moead_comparison_plot.sh raw_101w      # raw_101w DRL vs MOEA/D

cd /home/ltan/HHC

DRL_EXP="${1:-exp1}"
echo "=== MOEA/D vs DRL comparison (DRL exp: $DRL_EXP) ==="

# n=50
DRL_50="paretofront/pareto_results_50_${DRL_EXP}.pkl"
MOEAD_50="paretofront/moead_results_50_exp1.pkl"
if [ -f "$DRL_50" ] && [ -f "$MOEAD_50" ]; then
    python -u visualization/plot_moead_comparison.py \
        --graph_size 50 \
        --drl_path "$DRL_50" \
        --moead_path "$MOEAD_50" \
        --exp_name "comparison_${DRL_EXP}"
    echo "[OK] n=50 plot saved to images/comparison_${DRL_EXP}/moead_vs_drl_50.png"
else
    echo "[SKIP] n=50: DRL=$DRL_50 exists=$([ -f "$DRL_50" ] && echo yes || echo no), MOEA/D=$MOEAD_50 exists=$([ -f "$MOEAD_50" ] && echo yes || echo no)"
fi

# n=100
DRL_100="paretofront/pareto_results_100_${DRL_EXP}.pkl"
MOEAD_100="paretofront/moead_results_100_exp1.pkl"
if [ -f "$DRL_100" ] && [ -f "$MOEAD_100" ]; then
    python -u visualization/plot_moead_comparison.py \
        --graph_size 100 \
        --drl_path "$DRL_100" \
        --moead_path "$MOEAD_100" \
        --exp_name "comparison_${DRL_EXP}"
    echo "[OK] n=100 plot saved to images/comparison_${DRL_EXP}/moead_vs_drl_100.png"
else
    echo "[SKIP] n=100: DRL=$DRL_100 exists=$([ -f "$DRL_100" ] && echo yes || echo no), MOEA/D=$MOEAD_100 exists=$([ -f "$MOEAD_100" ] && echo yes || echo no)"
fi

echo "=== Done ==="
