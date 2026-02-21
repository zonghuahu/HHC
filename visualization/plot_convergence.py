"""
Multi-Lambda Convergence Visualization Script
Loads saved checkpoints and evaluates at 5 λ values to produce convergence curves.

Usage:
  python visualization/plot_convergence.py              # run evaluation + plot
  python visualization/plot_convergence.py --plot-only   # plot from cached results
"""
import os
import sys
import json
import argparse
import pickle
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nets.attention_model import AttentionModel, set_decode_type
from train import rollout, get_inner_model
from utils import torch_load_cpu, load_problem, move_to

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ===== Configuration =====
LAMBDAS = [(0.0, 1.0), (0.3, 0.7), (0.5, 0.5), (0.7, 0.3), (1.0, 0.0)]
COLORS = ['#E91E63', '#FF9800', '#4CAF50', '#2196F3', '#9C27B0']
LABELS = [f'λ=({l1},{l2})' for l1, l2 in LAMBDAS]

EXPERIMENTS = {
    'n=50': {
        'graph_size': 50,
        'checkpoint_sources': [
            # (directory, epoch_start, epoch_end_exclusive)
            ('outputs/agh_50/run_20260213T165016', 0, 54),
            ('outputs/agh_50/run_20260213T214438', 54, 100),
        ],
    },
    'n=100': {
        'graph_size': 100,
        'checkpoint_sources': [
            ('outputs/agh_100/run_20260214T095740', 0, 100),
        ],
    },
}

VAL_SIZE = 1000
EVAL_BATCH_SIZE = 100
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'paretofront')


def load_model(checkpoint_path, graph_size, device):
    """Load model from checkpoint."""
    problem = load_problem('agh')
    load_data = torch_load_cpu(checkpoint_path)

    model = AttentionModel(
        embedding_dim=128,
        hidden_dim=128,
        problem=problem,
        n_encode_layers=3,
        mask_inner=True,
        mask_logits=True,
        normalization='batch',
        tanh_clipping=10.,
        checkpoint_encoder=False,
        shrink_size=None,
        wo_time=False,
        rnn_time=False,
    ).to(device)

    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})
    return model


def evaluate_checkpoint(model, val_dataset, lambda_pair, device):
    """Evaluate model at a specific λ value. Returns avg scalarized cost."""
    set_decode_type(model, "greedy")
    model.eval()

    l1, l2 = lambda_pair

    # Create opts-like object for rollout
    class Opts:
        pass
    opts = Opts()
    opts.device = device
    opts.eval_batch_size = EVAL_BATCH_SIZE
    opts.no_progress_bar = True

    # Build lambda_vector matching eval batch size
    # rollout handles batching internally, so we pass lambda_vector as a fixed tensor
    # that will be expanded inside rollout
    lambda_vec = torch.tensor([[l1, l2]], dtype=torch.float, device=device)

    cost = rollout(model, val_dataset, opts, lambda_vector=lambda_vec)
    cost = cost.sum(1)  # sum across fleets
    return cost.mean().item()


def run_evaluation(device):
    """Evaluate all checkpoints at all λ values. Returns results dict."""
    problem = load_problem('agh')
    results = {}

    for exp_name, exp_config in EXPERIMENTS.items():
        print(f"\n{'='*60}")
        print(f"  Evaluating {exp_name} (graph_size={exp_config['graph_size']})")
        print(f"{'='*60}")

        graph_size = exp_config['graph_size']

        # Generate fixed validation dataset
        torch.manual_seed(4321)
        np.random.seed(4321)
        val_dataset = problem.make_dataset(size=graph_size, num_samples=VAL_SIZE)

        n_epochs = sum(end - start for _, start, end in exp_config['checkpoint_sources'])
        # results[exp_name][lambda_idx] = list of costs per epoch
        results[exp_name] = {li: [] for li in range(len(LAMBDAS))}

        for src_dir, ep_start, ep_end in exp_config['checkpoint_sources']:
            src_dir_full = os.path.join(os.path.dirname(__file__), '..', src_dir)
            for epoch in range(ep_start, ep_end):
                ckpt_path = os.path.join(src_dir_full, f'epoch-{epoch}.pt')
                if not os.path.exists(ckpt_path):
                    print(f"  [WARN] Missing checkpoint: {ckpt_path}")
                    for li in range(len(LAMBDAS)):
                        results[exp_name][li].append(float('nan'))
                    continue

                model = load_model(ckpt_path, graph_size, device)
                display_epoch = epoch - exp_config['checkpoint_sources'][0][1]  # relative epoch

                for li, lam in enumerate(LAMBDAS):
                    cost = evaluate_checkpoint(model, val_dataset, lam, device)
                    results[exp_name][li].append(cost)

                print(f"  Epoch {display_epoch:3d} | " +
                      " | ".join(f"λ={LAMBDAS[li]}: {results[exp_name][li][-1]:.1f}" for li in range(len(LAMBDAS))))

                # Free memory
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return results


def plot_results(results, save_path):
    """Plot convergence curves: left=n50, right=n100."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, (exp_name, exp_data) in enumerate(results.items()):
        ax = axes[ax_idx]

        for li in range(len(LAMBDAS)):
            costs = exp_data[li]
            epochs = list(range(len(costs)))
            l1, l2 = LAMBDAS[li]

            ax.plot(epochs, costs, '-', color=COLORS[li], linewidth=2,
                    label=LABELS[li], alpha=0.85)

            # Mark best epoch
            best_idx = np.nanargmin(costs)
            best_cost = costs[best_idx]
            ax.scatter([best_idx], [best_cost], c=COLORS[li], s=120, marker='*',
                       zorder=5, edgecolors='black', linewidths=0.5)

        ax.set_xlabel('Epoch', fontsize=13)
        ax.set_ylabel('Validation Cost (λ₁·f₁ + λ₂·f₂)', fontsize=13)
        ax.set_title(f'{exp_name}', fontsize=15, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Training Convergence at Different λ Weights', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved plot to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Multi-lambda convergence plot')
    parser.add_argument('--plot-only', action='store_true', help='Plot from cached results (skip evaluation)')
    parser.add_argument('--save', type=str, default=None, help='Output image path')
    parser.add_argument('--no-cuda', action='store_true', help='Force CPU')
    args = parser.parse_args()

    save_path = args.save or os.path.join(CACHE_DIR, 'convergence.png')
    cache_path = os.path.join(CACHE_DIR, 'convergence_data.pkl')

    if args.plot_only:
        if not os.path.exists(cache_path):
            print("[ERROR] No cached results found. Run without --plot-only first.")
            return
        with open(cache_path, 'rb') as f:
            results = pickle.load(f)
        print(f"[INFO] Loaded cached results from {cache_path}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
        print(f"[INFO] Using device: {device}")

        results = run_evaluation(device)

        # Cache results
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"[INFO] Cached results to {cache_path}")

    plot_results(results, save_path)


if __name__ == '__main__':
    main()
