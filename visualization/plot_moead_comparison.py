# -*- coding: utf-8 -*-
"""
Plot Pareto front comparison: MOEA/D vs DRL baseline.
Overlays both fronts for visual comparison.
"""
import os
import pickle
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_results(filepath):
    """Load Pareto results from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def plot_comparison(drl_path, moead_path, graph_size, output_dir='paretofront'):
    """Plot DRL vs MOEA/D Pareto fronts."""
    drl_results = load_results(drl_path)
    moead_results = load_results(moead_path)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # DRL
    drl_f1 = [r['f1_mean'] for r in drl_results]
    drl_f2 = [r['f2_mean'] for r in drl_results]
    ax.plot(drl_f1, drl_f2, 'o-', color='#2196F3', markersize=8,
            linewidth=2, label='DRL (Ours)', zorder=3)

    # MOEA/D
    moead_f1 = [r['f1_mean'] for r in moead_results]
    moead_f2 = [r['f2_mean'] for r in moead_results]
    ax.plot(moead_f1, moead_f2, 's--', color='#FF5722', markersize=8,
            linewidth=2, label='MOEA/D', zorder=3)

    # Add lambda labels
    for r in drl_results:
        l1 = r['lambda'][0]
        ax.annotate(f'λ₁={l1:.1f}', (r['f1_mean'], r['f2_mean']),
                    textcoords="offset points", xytext=(5, 8),
                    fontsize=7, color='#2196F3', alpha=0.7)

    ax.set_xlabel('f₁: Total Distance', fontsize=13)
    ax.set_ylabel('f₂: Total Waiting Time', fontsize=13)
    ax.set_title(f'Pareto Front Comparison — HHCRSP{graph_size}',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir,
                            f'moead_vs_drl_{graph_size}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {out_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_size', type=int, default=50)
    parser.add_argument('--drl_path', type=str, default=None,
                        help='Path to DRL pareto results .pkl')
    parser.add_argument('--moead_path', type=str, default=None,
                        help='Path to MOEA/D pareto results .pkl')
    parser.add_argument('--output_dir', type=str, default='paretofront')
    opts = parser.parse_args()

    if opts.drl_path is None:
        opts.drl_path = f'paretofront/pareto_results_{opts.graph_size}_exp1.pkl'
    if opts.moead_path is None:
        opts.moead_path = f'paretofront/moead_results_{opts.graph_size}_exp1.pkl'

    plot_comparison(opts.drl_path, opts.moead_path, opts.graph_size,
                    opts.output_dir)
