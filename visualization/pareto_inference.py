"""
DRL Pareto front inference: sweep 101 lambda weight vectors on shared test data.

For each lambda = (l1, l2) with l1 in [0, 0.01, ..., 1.0]:
  - Run greedy decoding across all 6 fleets
  - Record mean/std of f1 (distance) and f2 (waiting time)

Output: paretofront/pareto_results_{graph_size}_{exp_name}.pkl
        paretofront/pareto_front_{graph_size}_{exp_name}.png

Usage:
  python -u visualization/pareto_inference.py \
      --load_path outputs/agh_50/run_xxx/epoch-99.pt \
      --graph_size 50 \
      --val_dataset paretofront/shared_test_50.pkl \
      --exp_name exp1 \
      --output_dir paretofront
"""
import os
import sys
import argparse
import pickle
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.functions import load_model
from utils import move_to
from nets.attention_model import set_decode_type

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def pareto_inference(model, dataset, device, num_lambdas=101):
    """Run inference across a grid of lambda values, return per-lambda results."""
    set_decode_type(model, "greedy")
    model.eval()

    lambdas = np.linspace(0, 1, num_lambdas)
    results = []

    for idx, l1 in enumerate(lambdas):
        l2 = 1.0 - l1
        lv = torch.tensor([[l1, l2]], dtype=torch.float32, device=device)

        all_f1, all_f2 = [], []

        for bat in torch.utils.data.DataLoader(dataset, batch_size=200):
            bs = bat['loc'].size(0)
            lv_batch = lv.expand(bs, -1)

            bat_tw_left = bat['arrival'].repeat(
                len(model.fleet_info['next_duration']) + 1, 1, 1).to(device)
            bat_tw_right = bat['departure']
            need = bat['need']

            batch_f1 = torch.zeros(bs, device=device)
            batch_f2 = torch.zeros(bs, device=device)

            for f in model.fleet_info['order']:
                next_dur = torch.tensor(
                    model.fleet_info['next_duration'][model.fleet_info['precedence'][f]]
                ).repeat(bs, 1).type_as(bat['loc'])

                tw_right = bat_tw_right - torch.gather(next_dur, 1, bat['type'])
                tw_right = torch.cat((torch.full_like(tw_right[:, :1], 1441), tw_right), dim=1)

                tw_left = bat_tw_left[model.fleet_info['precedence'][f]]
                tw_left = torch.cat((torch.zeros_like(tw_left[:, :1]), tw_left), dim=1)

                duration = torch.tensor(
                    model.fleet_info['duration'][f]
                ).repeat(bs, 1).type_as(bat['loc'])

                if f == 1:
                    fmask = (need == 1) | (need == 9)
                elif f == 2:
                    fmask = (need == 2) | (need == 7)
                elif f == 3:
                    fmask = (need == 3) | (need == 7)
                elif f == 4:
                    fmask = (need == 4) | (need == 8)
                elif f == 5:
                    fmask = (need == 5) | (need == 8)
                elif f == 6:
                    fmask = (need == 6) | (need == 9)
                else:
                    fmask = (need == f)

                tw_right_f = tw_right.clone()
                tw_right_f[:, 1:] = tw_right[:, 1:] * fmask.type_as(tw_right).float()
                tw_left_f = tw_left.clone()
                tw_left_f[:, 1:] = tw_left[:, 1:] * fmask.type_as(tw_left).float()
                need_f = need.clone() * fmask.type_as(need).float()

                fleet_bat = {
                    'loc': bat['loc'],
                    'distance': model.distance.expand(bs, len(model.distance)),
                    'duration': torch.gather(duration, 1, bat['type']),
                    'tw_right': tw_right_f,
                    'tw_left': tw_left_f,
                    'fleet': torch.full((bs, 1), f - 1).type_as(bat['loc']),
                    'need': need_f,
                }

                if model.rnn_time:
                    model.pre_tw = None

                with torch.no_grad():
                    f1, f2, _, serve_time = model(move_to(fleet_bat, device), lambda_vector=lv_batch)

                batch_f1 += f1
                batch_f2 += f2

                next_stage = model.fleet_info['precedence'][f] + 1
                fmask = fmask.to(device)
                if f == 1:
                    bat_tw_left[next_stage] = torch.where(fmask, serve_time[:, 1:], bat_tw_left[next_stage])
                else:
                    bat_tw_left[next_stage] = torch.where(fmask, serve_time[:, 1:] + 10, bat_tw_left[next_stage])

            all_f1.append(batch_f1.cpu())
            all_f2.append(batch_f2.cpu())

        all_f1 = torch.cat(all_f1)
        all_f2 = torch.cat(all_f2)

        r = {
            'lambda': (l1, l2),
            'f1_mean': all_f1.mean().item(),
            'f1_std': all_f1.std().item(),
            'f2_mean': all_f2.mean().item(),
            'f2_std': all_f2.std().item(),
        }
        results.append(r)

        if idx % 10 == 0 or idx == num_lambdas - 1:
            print(f"  [{idx+1}/{num_lambdas}] λ=({l1:.2f},{l2:.2f}) "
                  f"f1={r['f1_mean']:.1f}±{r['f1_std']:.1f}  "
                  f"f2={r['f2_mean']:.1f}±{r['f2_std']:.1f}")

    return results


def plot_pareto(results, save_path, graph_size, exp_name):
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
    ax.set_title(f'DRL Pareto Front: n={graph_size} ({exp_name})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Pareto front plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, required=True)
    parser.add_argument('--graph_size', type=int, default=50)
    parser.add_argument('--val_size', type=int, default=1000)
    parser.add_argument('--val_dataset', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--exp_name', type=str, default='exp1')
    parser.add_argument('--output_dir', type=str, default='paretofront')
    parser.add_argument('--num_lambdas', type=int, default=101)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model from {args.load_path} ...")
    model, model_args = load_model(args.load_path)
    model = model.to(device)

    from problems.agh.problem_agh import AGH
    if args.val_dataset:
        dataset = AGH.make_dataset(filename=args.val_dataset, size=args.graph_size, num_samples=args.val_size)
    else:
        dataset = AGH.make_dataset(size=args.graph_size, num_samples=args.val_size)

    print(f"Dataset: {len(dataset)} instances, n={args.graph_size}")
    print(f"Sweeping {args.num_lambdas} lambda values ...")

    results = pareto_inference(model, dataset, device, num_lambdas=args.num_lambdas)

    os.makedirs(args.output_dir, exist_ok=True)

    pkl_path = os.path.join(args.output_dir, f'pareto_results_{args.graph_size}_{args.exp_name}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"[OK] Results saved to: {pkl_path}")

    txt_path = os.path.join(args.output_dir, f'pareto_results_{args.graph_size}_{args.exp_name}.txt')
    with open(txt_path, 'w') as f:
        f.write(f"{'lambda1':>8s} {'lambda2':>8s} {'f1_mean':>10s} {'f1_std':>10s} {'f2_mean':>10s} {'f2_std':>10s}\n")
        for r in results:
            f.write(f"{r['lambda'][0]:8.4f} {r['lambda'][1]:8.4f} "
                    f"{r['f1_mean']:10.2f} {r['f1_std']:10.2f} "
                    f"{r['f2_mean']:10.2f} {r['f2_std']:10.2f}\n")
    print(f"[OK] Text results saved to: {txt_path}")

    png_path = os.path.join(args.output_dir, f'pareto_front_{args.graph_size}_{args.exp_name}.png')
    plot_pareto(results, png_path, args.graph_size, args.exp_name)


if __name__ == '__main__':
    main()
