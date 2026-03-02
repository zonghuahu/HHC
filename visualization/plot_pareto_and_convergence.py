"""
Visualization: convergence curves and Pareto front plots for WE-Add DRL.

Usage:
  # Plot convergence only (from validate_log.txt)
  python visualization/plot_pareto_and_convergence.py --convergence --size 50

  # Plot Pareto front only (from .pkl results)
  python visualization/plot_pareto_and_convergence.py --pareto --size 50

  # Plot both for n=50 and n=100
  python visualization/plot_pareto_and_convergence.py --all
"""
import os
import re
import sys
import argparse
import pickle
import glob
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'images')


def find_run_dir(size, prefix='train_weadd'):
    """Find the latest run directory matching the prefix."""
    pattern = os.path.join(PROJECT_ROOT, f'outputs/agh_{size}/{prefix}_*')
    dirs = sorted(glob.glob(pattern))
    return dirs[-1] if dirs else None


def parse_validate_log(log_path):
    """Parse validate_log.txt, return (epochs, costs)."""
    epochs, costs = [], []
    pattern = re.compile(r'Validating Epoch (\d+), Validation avg_cost: ([\d.eE+-]+)')
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                costs.append(float(m.group(2)))
    return epochs, costs


def plot_convergence(size, save_dir, run_dir=None):
    """Plot validation cost convergence."""
    if run_dir is None:
        run_dir = find_run_dir(size)
    if run_dir is None:
        print(f"[SKIP] No run directory found for n={size}")
        return False

    log_path = os.path.join(run_dir, 'validate_log.txt')
    if not os.path.exists(log_path):
        print(f"[SKIP] {log_path} not found")
        return False

    epochs, costs = parse_validate_log(log_path)
    if not epochs:
        print(f"[SKIP] No data in {log_path}")
        return False

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, costs, '-', color='#2196F3', linewidth=2,
            label='Validation cost (λ=0.5, 0.5)')

    best_idx = int(np.argmin(costs))
    ax.scatter([epochs[best_idx]], [costs[best_idx]], c='#E91E63', s=150,
               marker='*', zorder=5, edgecolors='black', linewidths=0.5,
               label=f'Best: Epoch {epochs[best_idx]} ({costs[best_idx]:.1f})')

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Validation Cost (λ₁·f₁ + λ₂·f₂)', fontsize=13)
    ax.set_title(f'WE-Add DRL n={size} — Training Convergence', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, f'convergence_weadd_{size}.png')
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Convergence plot: {save_path}")
    return True


def plot_pareto(size, save_dir, exp_name='weadd'):
    """Plot Pareto front from .pkl results."""
    pkl_path = os.path.join(PROJECT_ROOT, 'paretofront', f'pareto_results_{size}_{exp_name}.pkl')
    if not os.path.exists(pkl_path):
        print(f"[SKIP] {pkl_path} not found")
        return False

    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)

    f1_means = [r['f1_mean'] for r in results]
    f2_means = [r['f2_mean'] for r in results]
    lambdas = [r['lambda'][0] for r in results]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    scatter = ax.scatter(f1_means, f2_means, c=lambdas,
                         cmap='coolwarm', s=80, zorder=5,
                         edgecolors='black', linewidths=0.5)

    sorted_idx = np.argsort(f1_means)
    ax.plot([f1_means[i] for i in sorted_idx],
            [f2_means[i] for i in sorted_idx],
            'k--', alpha=0.3, linewidth=1)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('λ₁ (Distance Weight)', fontsize=11)
    ax.set_xlabel('Total Travel Distance (f₁)', fontsize=13)
    ax.set_ylabel('Total Patient Waiting Time (f₂)', fontsize=13)
    ax.set_title(f'DRL Pareto Front: n={size} ({exp_name})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, f'pareto_front_weadd_{size}.png')
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Pareto front plot: {save_path}")
    return True


def plot_pareto_comparison(sizes, save_dir, exp_name='weadd'):
    """Plot Pareto fronts for multiple sizes on one figure."""
    fig, axes = plt.subplots(1, len(sizes), figsize=(10 * len(sizes), 7))
    if len(sizes) == 1:
        axes = [axes]

    for ax, size in zip(axes, sizes):
        pkl_path = os.path.join(PROJECT_ROOT, 'paretofront', f'pareto_results_{size}_{exp_name}.pkl')
        if not os.path.exists(pkl_path):
            ax.set_title(f'n={size} (no data)')
            continue

        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)

        f1_means = [r['f1_mean'] for r in results]
        f2_means = [r['f2_mean'] for r in results]
        lambdas = [r['lambda'][0] for r in results]

        scatter = ax.scatter(f1_means, f2_means, c=lambdas,
                             cmap='coolwarm', s=80, zorder=5,
                             edgecolors='black', linewidths=0.5)
        sorted_idx = np.argsort(f1_means)
        ax.plot([f1_means[i] for i in sorted_idx],
                [f2_means[i] for i in sorted_idx],
                'k--', alpha=0.3, linewidth=1)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('λ₁', fontsize=10)
        ax.set_xlabel('f₁ (Distance)', fontsize=12)
        ax.set_ylabel('f₂ (Waiting)', fontsize=12)
        ax.set_title(f'n={size}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'DRL Pareto Fronts ({exp_name})', fontsize=15, fontweight='bold')
    save_path = os.path.join(save_dir, f'pareto_comparison_{exp_name}.png')
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Pareto comparison: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--convergence', action='store_true')
    parser.add_argument('--pareto', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--size', type=int, nargs='+', default=[50, 100])
    parser.add_argument('--exp_name', type=str, default='weadd')
    parser.add_argument('--save_dir', type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.all:
        args.convergence = True
        args.pareto = True

    if not args.convergence and not args.pareto:
        args.convergence = True
        args.pareto = True

    for s in args.size:
        if args.convergence:
            plot_convergence(s, args.save_dir)
        if args.pareto:
            plot_pareto(s, args.save_dir, args.exp_name)

    if args.pareto and len(args.size) > 1:
        plot_pareto_comparison(args.size, args.save_dir, args.exp_name)


if __name__ == '__main__':
    main()
