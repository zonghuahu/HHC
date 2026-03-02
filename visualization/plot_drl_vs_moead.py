"""
Plot DRL vs MOEA/D Pareto front comparison — side-by-side for n=50 and n=100.

Reads:
  - DRL results:   paretofront/pareto_results_{size}_weadd.pkl
  - MOEA/D results: scripts/moead/moead_results_{size}_exp1.pkl

Output:
  - images/pareto_drl_vs_moead.png

Usage:
  python -u visualization/plot_drl_vs_moead.py
  python -u visualization/plot_drl_vs_moead.py --drl_exp weadd --moead_dir scripts/moead
"""
import os
import sys
import argparse
import pickle
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def load_results(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def plot_comparison(drl_50, moead_50, drl_100, moead_100, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    label_lambdas = {0.0, 0.3, 0.5, 0.7, 1.0}

    for ax, drl, moead, size in [(ax1, drl_50, moead_50, 50),
                                  (ax2, drl_100, moead_100, 100)]:

        # --- DRL ---
        drl_f1 = [r['f1_mean'] for r in drl]
        drl_f2 = [r['f2_mean'] for r in drl]
        drl_l1 = [r['lambda'][0] if isinstance(r['lambda'], (list, tuple)) else r['lambda'] for r in drl]

        sorted_idx = np.argsort(drl_f1)
        ax.plot([drl_f1[i] for i in sorted_idx],
                [drl_f2[i] for i in sorted_idx],
                '-', color='gray', alpha=0.4, linewidth=1, zorder=1)

        sc_drl = ax.scatter(drl_f1, drl_f2, c=drl_l1, cmap='coolwarm',
                            vmin=0, vmax=1, s=90, zorder=5,
                            edgecolors='black', linewidths=0.6,
                            marker='o', label='DRL (Ours)')

        for i, l1 in enumerate(drl_l1):
            rounded = round(l1, 1)
            if rounded in label_lambdas:
                ax.annotate(f'λ={rounded}', (drl_f1[i], drl_f2[i]),
                            textcoords='offset points', xytext=(-25, 8),
                            fontsize=7.5, color='#555555', fontstyle='italic')

        # --- MOEA/D ---
        moead_f1 = [r['f1_mean'] for r in moead]
        moead_f2 = [r['f2_mean'] for r in moead]
        moead_l1 = [r['lambda'][0] if isinstance(r['lambda'], (list, tuple)) else r['lambda'] for r in moead]

        sorted_idx_m = np.argsort(moead_f1)
        ax.plot([moead_f1[i] for i in sorted_idx_m],
                [moead_f2[i] for i in sorted_idx_m],
                '--', color='gray', alpha=0.4, linewidth=1, zorder=1)

        ax.scatter(moead_f1, moead_f2, c=moead_l1, cmap='coolwarm',
                   vmin=0, vmax=1, s=90, zorder=4,
                   edgecolors='black', linewidths=0.6,
                   marker='s', label='MOEA/D')

        ax.set_xlabel('Total Travel Distance (f₁)', fontsize=13)
        ax.set_ylabel('Total Patient Waiting Time (f₂)', fontsize=13)
        ax.set_title(f'Pareto Front Comparison — HHCRSP{size}',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.25)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markeredgecolor='black', markersize=10, label='DRL (Ours)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                   markeredgecolor='black', markersize=10, label='MOEA/D'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
                  framealpha=0.9)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('λ₁ (Distance Weight)', fontsize=11)

    plt.subplots_adjust(left=0.06, right=0.90, wspace=0.25)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drl_exp', type=str, default='weadd',
                        help='DRL experiment name (for .pkl filename)')
    parser.add_argument('--moead_dir', type=str, default='scripts/moead',
                        help='Directory containing MOEA/D results')
    parser.add_argument('--save_path', type=str,
                        default=os.path.join(PROJECT_ROOT, 'images', 'pareto_drl_vs_moead.png'))
    args = parser.parse_args()

    drl_50_path = os.path.join(PROJECT_ROOT, 'paretofront', f'pareto_results_50_{args.drl_exp}.pkl')
    drl_100_path = os.path.join(PROJECT_ROOT, 'paretofront', f'pareto_results_100_{args.drl_exp}.pkl')
    moead_50_path = os.path.join(PROJECT_ROOT, args.moead_dir, 'moead_results_50_exp1.pkl')
    moead_100_path = os.path.join(PROJECT_ROOT, args.moead_dir, 'moead_results_100_exp1.pkl')

    missing = []
    for name, path in [('DRL n=50', drl_50_path), ('DRL n=100', drl_100_path),
                        ('MOEA/D n=50', moead_50_path), ('MOEA/D n=100', moead_100_path)]:
        if not os.path.exists(path):
            missing.append(f"  {name}: {path}")
        else:
            print(f"[OK] Found: {path}")

    if missing:
        print("\n[WARN] Missing result files:")
        for m in missing:
            print(m)
        print("\nWill plot with available data only.\n")

    drl_50 = load_results(drl_50_path) if os.path.exists(drl_50_path) else None
    drl_100 = load_results(drl_100_path) if os.path.exists(drl_100_path) else None
    moead_50 = load_results(moead_50_path) if os.path.exists(moead_50_path) else None
    moead_100 = load_results(moead_100_path) if os.path.exists(moead_100_path) else None

    if drl_50 is None and moead_50 is None and drl_100 is None and moead_100 is None:
        print("[ERROR] No result files found. Run DRL pareto inference and MOEA/D first.")
        sys.exit(1)

    if drl_50 is None:
        drl_50 = []
    if drl_100 is None:
        drl_100 = []
    if moead_50 is None:
        moead_50 = []
    if moead_100 is None:
        moead_100 = []

    plot_comparison(drl_50, moead_50, drl_100, moead_100, args.save_path)


if __name__ == '__main__':
    main()
