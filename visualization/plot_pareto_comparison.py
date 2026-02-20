"""
Pareto Front Comparison Script
Overlays Pareto fronts from multiple experiments on one plot.

Usage:
  python visualization/plot_pareto_comparison.py
  python visualization/plot_pareto_comparison.py --results pareto_results_50_exp1.pkl pareto_results_100_exp1.pkl
"""
import os
import argparse
import pickle
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_results(pkl_path):
    """Load Pareto results from pickle file."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def find_all_results():
    """Auto-find all pareto_results_*.pkl in paretofront/."""
    pattern = os.path.join(os.path.dirname(__file__), '..', 'paretofront', 'pareto_results_*.pkl')
    return sorted(glob.glob(pattern))


def plot_comparison(result_files, labels=None, save_path=None):
    """Plot multiple Pareto fronts on one chart."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
    markers = ['o', 's', 'D', '^', 'v']
    
    for i, pkl_path in enumerate(result_files):
        results = load_results(pkl_path)
        f1_means = [r['f1_mean'] for r in results]
        f2_means = [r['f2_mean'] for r in results]
        lambdas = [r['lambda'] for r in results]
        lambda_1_vals = [l[0] for l in lambdas]
        
        # Extract label from filename if not provided
        basename = os.path.basename(pkl_path)
        # e.g., pareto_results_50_exp1.pkl → n=50 (exp1)
        parts = basename.replace('pareto_results_', '').replace('.pkl', '').split('_')
        label = labels[i] if labels and i < len(labels) else f'n={parts[0]} ({parts[1]})'
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # Left plot: Pareto front
        ax = axes[0]
        ax.scatter(f1_means, f2_means, c=color, s=100, marker=marker, zorder=5,
                   edgecolors='black', linewidths=0.5, label=label)
        
        # Connect with line
        sorted_idx = np.argsort(f1_means)
        f1_sorted = [f1_means[j] for j in sorted_idx]
        f2_sorted = [f2_means[j] for j in sorted_idx]
        ax.plot(f1_sorted, f2_sorted, '--', color=color, alpha=0.4, linewidth=1.5)
        
        # Annotate extreme points
        ax.annotate(f'λ₁=0', (f1_means[0], f2_means[0]),
                    textcoords="offset points", xytext=(8, 8), fontsize=7, color=color)
        ax.annotate(f'λ₁=1', (f1_means[-1], f2_means[-1]),
                    textcoords="offset points", xytext=(8, -12), fontsize=7, color=color)
        
        # Right plot: f1 and f2 vs lambda
        ax2 = axes[1]
        ax2.plot(lambda_1_vals, f1_means, '-o', color=color, markersize=5,
                 linewidth=2, label=f'{label} - Distance (f₁)', alpha=0.8)
        
        # Create twin axis for f2
        if i == 0:
            ax2_twin = ax2.twinx()
        ax2_twin.plot(lambda_1_vals, f2_means, '--s', color=color, markersize=5,
                      linewidth=2, label=f'{label} - Waiting (f₂)', alpha=0.5)
    
    # Format left plot
    axes[0].set_xlabel('Total Travel Distance (f₁)', fontsize=13)
    axes[0].set_ylabel('Total Patient Waiting Time (f₂)', fontsize=13)
    axes[0].set_title('Pareto Front Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Format right plot
    axes[1].set_xlabel('λ₁ (Distance Weight)', fontsize=13)
    axes[1].set_ylabel('Travel Distance (f₁)', fontsize=13, color='#2196F3')
    ax2_twin.set_ylabel('Waiting Time (f₂)', fontsize=13, color='#FF5722')
    axes[1].set_title('Objectives vs. Preference Weight', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='center right')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), '..', 'paretofront', 'pareto_comparison.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved to: {save_path}")
    
    # Print comparison table
    print("\n" + "=" * 80)
    for i, pkl_path in enumerate(result_files):
        results = load_results(pkl_path)
        basename = os.path.basename(pkl_path)
        parts = basename.replace('pareto_results_', '').replace('.pkl', '').split('_')
        label = labels[i] if labels and i < len(labels) else f'n={parts[0]}'
        
        f1_range = (min(r['f1_mean'] for r in results), max(r['f1_mean'] for r in results))
        f2_range = (min(r['f2_mean'] for r in results), max(r['f2_mean'] for r in results))
        
        print(f"\n{label}:")
        print(f"  Distance range: {f1_range[0]:.1f} - {f1_range[1]:.1f} (Δ={f1_range[1]-f1_range[0]:.1f})")
        print(f"  Waiting range:  {f2_range[0]:.1f} - {f2_range[1]:.1f} (Δ={f2_range[1]-f2_range[0]:.1f})")
        print(f"  f₂ reduction:   {((f2_range[1]-f2_range[0])/f2_range[1]*100):.1f}% at cost of {((f1_range[1]-f1_range[0])/f1_range[0]*100):.1f}% more distance")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Compare Pareto fronts from multiple experiments')
    parser.add_argument('--results', nargs='*', help='Paths to pareto_results_*.pkl files')
    parser.add_argument('--labels', nargs='*', help='Labels for each result')
    parser.add_argument('--save', type=str, default=None, help='Output image path')
    args = parser.parse_args()
    
    if args.results:
        result_files = args.results
    else:
        result_files = find_all_results()
        if not result_files:
            print("[ERROR] No pareto_results_*.pkl files found in paretofront/")
            return
        print(f"[INFO] Found {len(result_files)} result files: {result_files}")
    
    plot_comparison(result_files, labels=args.labels, save_path=args.save)


if __name__ == '__main__':
    main()
