"""
Training Convergence Visualization Script
Plots validation cost over epochs for one or multiple runs.

Usage:
  python visualization/plot_convergence.py                          # auto-find latest
  python visualization/plot_convergence.py <validate_log.txt>       # single run
  python visualization/plot_convergence.py <log1> <log2> --labels "n=50" "n=100"  # compare
"""
import os
import re
import sys
import glob
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def parse_log(log_path):
    """Parse validate_log.txt file — handles both old and new formats."""
    with open(log_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Match: "####Validating Epoch X, Validation avg_cost: Y" or "Epoch X, Validation avg_cost: Y"
    pattern = r'Epoch (\d+), Validation avg_cost: ([\d.]+)'
    matches = re.findall(pattern, text)
    
    epochs = [int(m[0]) for m in matches]
    costs = [float(m[1]) for m in matches]
    
    return epochs, costs


def find_all_logs():
    """Find all validate_log.txt files."""
    pattern = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'agh_*', 'run_*', 'validate_log.txt')
    logs = glob.glob(pattern)
    return sorted(logs, key=os.path.getmtime)


def plot_convergence(log_paths, labels=None, save_path=None, max_epochs=100):
    """Plot convergence curves for one or multiple runs."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
    
    for i, log_spec in enumerate(log_paths):
        # Support merging: "log1.txt+log2.txt" merges into one curve
        if '+' in log_spec:
            all_epochs, all_costs = [], []
            for lp in log_spec.split('+'):
                ep, co = parse_log(lp.strip())
                all_epochs.extend(ep)
                all_costs.extend(co)
            # Sort by epoch and deduplicate
            paired = sorted(zip(all_epochs, all_costs))
            epochs = [p[0] for p in paired]
            costs = [p[1] for p in paired]
        else:
            epochs, costs = parse_log(log_spec)
        
        if not epochs:
            print(f"[WARNING] No data found in {log_spec}")
            continue
        
        # Truncate to max_epochs
        if len(epochs) > max_epochs:
            epochs = epochs[:max_epochs]
            costs = costs[:max_epochs]
        
        # Renumber epochs to be continuous 0, 1, 2, ...
        epochs = list(range(len(epochs)))
        
        label = labels[i] if labels and i < len(labels) else os.path.basename(os.path.dirname(log_spec.split('+')[0]))
        color = colors[i % len(colors)]
        
        # Main curve
        ax.plot(epochs, costs, '-', color=color, linewidth=2, label=label, alpha=0.8)
        ax.scatter(epochs, costs, c=color, s=15, alpha=0.3)
        
        # Smoothed curve
        if len(costs) > 10:
            window = min(10, len(costs) // 5)
            smoothed = np.convolve(costs, np.ones(window)/window, mode='valid')
            smooth_epochs = epochs[window-1:]
            ax.plot(smooth_epochs, smoothed, '--', color=color, linewidth=1.5, alpha=0.5)
        
        # Mark best epoch
        best_idx = np.argmin(costs)
        best_epoch = epochs[best_idx]
        best_cost = costs[best_idx]
        ax.scatter([best_epoch], [best_cost], c=color, s=150, marker='*', zorder=5,
                   edgecolors='black', linewidths=0.5)
        ax.annotate(f'Best: {best_cost:.1f} (ep.{best_epoch})',
                    xy=(best_epoch, best_cost),
                    textcoords="offset points", xytext=(10, 10), fontsize=9, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, alpha=0.7))
        
        print(f"[{label}] Epochs: {len(epochs)}, Start: {costs[0]:.2f}, "
              f"Best: {best_cost:.2f} (ep.{best_epoch}), Final: {costs[-1]:.2f}, "
              f"Improvement: {((costs[0] - best_cost) / costs[0] * 100):.1f}%")
    
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Validation Cost (Scalarized at λ=[0.5, 0.5])', fontsize=13)
    ax.set_title('Training Convergence', fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), '..', 'paretofront', 'convergence.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot training convergence curves')
    parser.add_argument('logs', nargs='*', help='Paths to validate_log.txt files')
    parser.add_argument('--labels', nargs='*', help='Labels for each log file')
    parser.add_argument('--save', type=str, default=None, help='Output image path')
    parser.add_argument('--max_epochs', type=int, default=100, help='Max number of epochs to show (default: 100)')
    args = parser.parse_args()
    
    if args.logs:
        log_paths = args.logs
    else:
        log_paths = find_all_logs()
        if not log_paths:
            print("[ERROR] No validate_log.txt files found!")
            return
        print(f"[INFO] Found {len(log_paths)} log files")
    
    plot_convergence(log_paths, labels=args.labels, save_path=args.save, max_epochs=args.max_epochs)


if __name__ == '__main__':
    main()
