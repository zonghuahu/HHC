# -*- coding: utf-8 -*-
"""
Plot Pareto front comparison: MOEA/D vs DRL.
Clean curves only, no bands, no labels.
"""
import os
import pickle
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_results(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def plot_comparison(drl_path, moead_path, graph_size, output_dir='paretofront'):
    drl_results = load_results(drl_path)
    moead_results = load_results(moead_path)

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            pass
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    # DRL: mean line only, solid, no markers
    drl_f1 = np.array([r['f1_mean'] for r in drl_results])
    drl_f2 = np.array([r['f2_mean'] for r in drl_results])
    sort_idx = np.argsort(drl_f1)
    ax.plot(drl_f1[sort_idx], drl_f2[sort_idx], '-', color='#4A90D9',
            linewidth=2.0, label='DRL (Ours)', zorder=3)

    # MOEA/D: dashed, square markers, markevery=3, markersize=5
    moead_f1 = np.array([r['f1_mean'] for r in moead_results])
    moead_f2 = np.array([r['f2_mean'] for r in moead_results])
    sort_m = np.argsort(moead_f1)
    ax.plot(moead_f1[sort_m], moead_f2[sort_m], 's--', color='#D94A4A',
            markersize=5, linewidth=1.8, markevery=3, label='MOEA/D', zorder=3)

    ax.set_xlabel(r'$f_1$: Total Travel Distance', fontsize=11)
    ax.set_ylabel(r'$f_2$: Total Patient Waiting Time', fontsize=11)
    ax.set_title(f'Pareto Front Comparison — HHCRSP n={graph_size}',
                fontsize=13, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle=':', alpha=0.2)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.8,
              edgecolor='lightgray')

    plt.tight_layout(pad=1.0)
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, f'moead_vs_drl_{graph_size}')
    for ext in ('png', 'pdf'):
        out_path = base + '.' + ext
        plt.savefig(out_path, dpi=200 if ext == 'png' else None,
                    bbox_inches='tight')
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
