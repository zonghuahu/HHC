"""
Plot Pareto front and convergence for agh_50 200 epoch training run.

Convergence: parsed from validate_log.txt (λ=0.5,0.5 validation cost).
Pareto front: requires running pareto_inference.py first to generate results.

Usage:
  # 1. Plot convergence (no model loading needed)
  python visualization/plot_agh50_200ep.py --convergence-only

  # 2. Generate Pareto results then plot (run pareto_inference first if needed)
  python visualization/pareto_inference.py --load_path outputs/agh_50/train_raw_50_200ep_20260228T162007/epoch-199.pt --graph_size 50 --exp_name raw_200ep
  python visualization/plot_agh50_200ep.py
"""
import os
import re
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')

RUN_DIR = 'outputs/agh_50/train_raw_50_200ep_20260228T162007'
VALIDATE_LOG = os.path.join(PROJECT_ROOT, RUN_DIR, 'validate_log.txt')
PARETO_PKL = os.path.join(PROJECT_ROOT, 'paretofront', 'pareto_results_50_raw_200ep.pkl')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'images', 'agh50_200ep')


def parse_validate_log(log_path):
    """Parse validate_log.txt, return (epochs, costs)."""
    epochs, costs = [], []
    pattern = re.compile(r'Validating Epoch (\d+), Validation avg_cost: ([\d.]+)')
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                costs.append(float(m.group(2)))
    return epochs, costs


def plot_convergence(save_path):
    """Plot validation cost convergence (λ=0.5, 0.5)."""
    if not os.path.exists(VALIDATE_LOG):
        print(f"[ERROR] {VALIDATE_LOG} not found")
        return False

    epochs, costs = parse_validate_log(VALIDATE_LOG)
    if not epochs:
        print("[ERROR] No data parsed from validate_log.txt")
        return False

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, costs, '-', color='#2196F3', linewidth=2, label='Validation cost (λ=0.5, 0.5)')

    best_idx = np.argmin(costs)
    ax.scatter([epochs[best_idx]], [costs[best_idx]], c='#E91E63', s=150, marker='*',
               zorder=5, edgecolors='black', linewidths=0.5, label=f'Best: Epoch {epochs[best_idx]}')

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Validation Cost (λ₁·f₁ + λ₂·f₂)', fontsize=13)
    ax.set_title('AGH n=50, 200 Epochs — Training Convergence', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Convergence plot saved to: {save_path}")
    return True


def plot_pareto_front(save_path):
    """Plot Pareto front from pareto_results_50_raw_200ep.pkl."""
    if not os.path.exists(PARETO_PKL):
        print(f"[INFO] Pareto results not found: {PARETO_PKL}")
        print("       Run first: python visualization/pareto_inference.py --load_path outputs/agh_50/train_raw_50_200ep_20260228T162007/epoch-199.pt --graph_size 50 --exp_name raw_200ep")
        return False

    with open(PARETO_PKL, 'rb') as f:
        results = pickle.load(f)

    f1_means = [r['f1_mean'] for r in results]
    f2_means = [r['f2_mean'] for r in results]
    lambdas = [r['lambda'] for r in results]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    scatter = ax.scatter(f1_means, f2_means, c=[l[0] for l in lambdas],
                         cmap='coolwarm', s=80, zorder=5, edgecolors='black', linewidths=0.5)

    sorted_idx = np.argsort(f1_means)
    f1_sorted = [f1_means[i] for i in sorted_idx]
    f2_sorted = [f2_means[i] for i in sorted_idx]
    ax.plot(f1_sorted, f2_sorted, 'k--', alpha=0.3, linewidth=1)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('λ₁ (Distance Weight)', fontsize=11)
    ax.set_xlabel('Total Travel Distance (f₁)', fontsize=13)
    ax.set_ylabel('Total Patient Waiting Time (f₂)', fontsize=13)
    ax.set_title('Pareto Front: AGH n=50, 200 Epochs (epoch-199)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Pareto front saved to: {save_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--convergence-only', action='store_true', help='Only plot convergence from validate_log')
    parser.add_argument('--pareto-only', action='store_true', help='Only plot Pareto front (requires .pkl)')
    parser.add_argument('--save-dir', type=str, default=OUTPUT_DIR, help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.pareto_only:
        plot_pareto_front(os.path.join(args.save_dir, 'pareto_front_50_200ep.png'))
    elif args.convergence_only:
        plot_convergence(os.path.join(args.save_dir, 'convergence_50_200ep.png'))
    else:
        ok1 = plot_convergence(os.path.join(args.save_dir, 'convergence_50_200ep.png'))
        ok2 = plot_pareto_front(os.path.join(args.save_dir, 'pareto_front_50_200ep.png'))
        if ok1 and ok2:
            print("\n[OK] Both plots saved to:", args.save_dir)


if __name__ == '__main__':
    main()
