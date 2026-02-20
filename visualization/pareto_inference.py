"""
Pareto Front Inference Script for WE-Add Multi-Objective HHC

Generate solutions for multiple weight vectors λ and plot the Pareto front
showing the trade-off between:
  f₁ = Total Travel Distance
  f₂ = Total Patient Waiting Time

Usage:
  python pareto_inference.py --load_path outputs/agh_50/<run_name>/epoch-X.pt
  python pareto_inference.py --load_path outputs/agh_50/<run_name>/epoch-X.pt --n_weights 21
  python pareto_inference.py --load_path outputs/agh_50/<run_name>/epoch-X.pt --val_dataset path/to/val.pkl
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import torch
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，确保服务器上也能保存图片
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from nets.attention_model import AttentionModel, set_decode_type
from utils import load_problem, move_to


def get_pareto_args():
    parser = argparse.ArgumentParser(description="Pareto Front Inference for WE-Add Multi-Objective HHC")
    parser.add_argument('--load_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--n_weights', type=int, default=11, help='Number of weight vectors (default: 11)')
    parser.add_argument('--graph_size', type=int, default=50, help='Problem graph size (default: 50)')
    parser.add_argument('--val_size', type=int, default=100, help='Number of validation instances (default: 100)')
    parser.add_argument('--val_dataset', type=str, default=None, help='Validation dataset .pkl file')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for inference')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save output files')
    parser.add_argument('--exp_name', type=str, default='exp1', help='Experiment name for output files (e.g., exp1, exp2)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    return parser.parse_args()


def load_model_from_checkpoint(load_path, device):
    """加载训练好的模型"""
    problem = load_problem('agh')
    model = AttentionModel(
        128,  # embedding_dim
        128,  # hidden_dim
        problem,
        n_encode_layers=3,
        mask_inner=True,
        mask_logits=True,
        normalization='batch',
        tanh_clipping=10.,
    ).to(device)

    # 加载检查点
    load_data = torch.load(load_path, map_location=device, weights_only=False)
    model.load_state_dict(load_data['model'])
    model.eval()
    set_decode_type(model, "greedy")

    print(f"Model loaded from {load_path}")
    return model, problem


def run_inference_with_lambda(model, dataset, lambda_vec, device, batch_size):
    """
    对给定的 λ 权重向量，使用贪婪解码生成解并计算两个目标值。
    返回每个实例的 (f1, f2) 值。
    """
    all_f1, all_f2 = [], []

    for bat in DataLoader(dataset, batch_size=batch_size):
        bat_f1, bat_f2 = [], []

        bat_tw_left = bat['arrival'].repeat(len(model.fleet_info['next_duration']) + 1, 1, 1).to(device)
        bat_tw_right = bat['departure']
        need = bat['need']

        # 创建 lambda_vector
        cur_batch_size = bat['loc'].size(0)
        lambda_vector = torch.tensor([lambda_vec], dtype=torch.float, device=device).expand(cur_batch_size, -1)

        fleet_f1_total = torch.zeros(cur_batch_size, device=device)
        fleet_f2_total = torch.zeros(cur_batch_size, device=device)

        for f in model.fleet_info['order']:
            next_duration = torch.tensor(
                model.fleet_info['next_duration'][model.fleet_info['precedence'][f]]
            ).repeat(bat['loc'].size(0), 1).type_as(bat['loc'])

            tw_right = bat_tw_right - torch.gather(next_duration, 1, bat['type'])
            tw_right = torch.cat((torch.full_like(tw_right[:, :1], 1441), tw_right), dim=1)

            tw_left = bat_tw_left[model.fleet_info['precedence'][f]]
            tw_left = torch.cat((torch.zeros_like(tw_left[:, :1]), tw_left), dim=1)

            duration = torch.tensor(model.fleet_info['duration'][f]) \
                .repeat(bat['loc'].size(0), 1).type_as(bat['loc'])

            if f == 1:
                mask = (need == 1) | (need == 9)
            elif f == 2:
                mask = (need == 2) | (need == 7)
            elif f == 3:
                mask = (need == 3) | (need == 7)
            elif f == 4:
                mask = (need == 4) | (need == 8)
            elif f == 5:
                mask = (need == 5) | (need == 8)
            elif f == 6:
                mask = (need == 6) | (need == 9)
            else:
                mask = (need == f)

            tw_right_filtered = tw_right.clone()
            tw_right_filtered[:, 1:] = tw_right[:, 1:] * mask.type_as(tw_right).float()

            tw_left_filtered = tw_left.clone()
            tw_left_filtered[:, 1:] = tw_left[:, 1:] * mask.type_as(tw_left).float()

            need_filtered = need.clone()
            need_filtered = need_filtered * mask.type_as(need).float()

            fleet_bat = {
                'loc': bat['loc'],
                'distance': model.distance.expand(bat['loc'].size(0), len(model.distance)),
                'duration': torch.gather(duration, 1, bat['type']),
                'tw_right': tw_right_filtered,
                'tw_left': tw_left_filtered,
                'fleet': torch.full((bat['loc'].size(0), 1), f - 1).type_as(bat['loc']),
                'need': need_filtered,
            }

            with torch.no_grad():
                _, _, serve_time, f1, f2 = model(
                    move_to(fleet_bat, device),
                    lambda_vector=lambda_vector
                )

            fleet_f1_total += f1
            fleet_f2_total += f2

            # 更新时间窗口
            next_stage = model.fleet_info['precedence'][f] + 1
            mask = mask.to(device)
            if f == 1:
                bat_tw_left[next_stage] = torch.where(mask, serve_time[:, 1:], bat_tw_left[next_stage])
            else:
                bat_tw_left[next_stage] = torch.where(mask, serve_time[:, 1:] + 10, bat_tw_left[next_stage])

        all_f1.append(fleet_f1_total.cpu())
        all_f2.append(fleet_f2_total.cpu())

    return torch.cat(all_f1, 0), torch.cat(all_f2, 0)


def plot_pareto_front(results, output_path, graph_size=50, exp_name='exp1'):
    """绘制 Pareto 前沿图"""
    f1_means = [r['f1_mean'] for r in results]
    f2_means = [r['f2_mean'] for r in results]
    lambdas = [r['lambda'] for r in results]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # 绘制散点
    scatter = ax.scatter(f1_means, f2_means, c=[l[0] for l in lambdas],
                         cmap='coolwarm', s=120, zorder=5, edgecolors='black', linewidths=0.5)

    # 连接线
    sorted_indices = np.argsort(f1_means)
    f1_sorted = [f1_means[i] for i in sorted_indices]
    f2_sorted = [f2_means[i] for i in sorted_indices]
    ax.plot(f1_sorted, f2_sorted, 'k--', alpha=0.3, linewidth=1)

    # 标注 λ 值
    for i, (f1, f2, lam) in enumerate(zip(f1_means, f2_means, lambdas)):
        ax.annotate(f'λ₁={lam[0]:.1f}', (f1, f2),
                    textcoords="offset points", xytext=(8, 8), fontsize=7, alpha=0.8)

    # 颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('λ₁ (Distance Weight)', fontsize=11)

    ax.set_xlabel('Total Travel Distance (f₁)', fontsize=13)
    ax.set_ylabel('Total Patient Waiting Time (f₂)', fontsize=13)
    ax.set_title(f'Pareto Front: Travel Distance vs. Waiting Time (n={graph_size}, {exp_name})', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Pareto front saved to {output_path}")


def main():
    args = get_pareto_args()

    # 设置设备和随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # 加载模型
    model, problem = load_model_from_checkpoint(args.load_path, device)

    # 加载或生成数据集
    dataset = problem.make_dataset(
        size=args.graph_size,
        num_samples=args.val_size,
        filename=args.val_dataset
    )
    print(f"Dataset: {len(dataset)} instances, graph_size={args.graph_size}")

    # === 生成权重向量并推理 ===
    weight_vectors = []
    for i in range(args.n_weights):
        lambda_1 = i / (args.n_weights - 1)  # 0.0, 0.1, ..., 1.0
        lambda_2 = 1 - lambda_1
        weight_vectors.append([lambda_1, lambda_2])

    results = []
    print(f"\nRunning inference with {args.n_weights} weight vectors...")
    for lambda_vec in tqdm(weight_vectors, desc="Weight vectors"):
        f1_vals, f2_vals = run_inference_with_lambda(model, dataset, lambda_vec, device, args.batch_size)
        result = {
            'lambda': lambda_vec,
            'f1_mean': f1_vals.mean().item(),
            'f2_mean': f2_vals.mean().item(),
            'f1_std': f1_vals.std().item(),
            'f2_std': f2_vals.std().item(),
            'f1_all': f1_vals.numpy(),
            'f2_all': f2_vals.numpy(),
        }
        results.append(result)
        print(f"  λ={lambda_vec} → f₁={result['f1_mean']:.2f} ± {result['f1_std']:.2f}, "
              f"f₂={result['f2_mean']:.2f} ± {result['f2_std']:.2f}")

    # === 绘制 Pareto 前沿 ===
    # Save to HHC/paretofront/ directory with graph size and experiment name
    pareto_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paretofront')
    os.makedirs(pareto_dir, exist_ok=True)
    
    filename = f'pareto_front_{args.graph_size}_{args.exp_name}.png'
    output_path = os.path.join(pareto_dir, filename)
    plot_pareto_front(results, output_path, graph_size=args.graph_size, exp_name=args.exp_name)

    # Also save a copy in the run output_dir for reference
    output_path_run = os.path.join(args.output_dir, filename)
    plot_pareto_front(results, output_path_run, graph_size=args.graph_size, exp_name=args.exp_name)

    # === 保存结果数据 ===
    results_filename = f'pareto_results_{args.graph_size}_{args.exp_name}.pkl'
    results_path = os.path.join(pareto_dir, results_filename)
    save_results = [{k: v for k, v in r.items() if k != 'f1_all' and k != 'f2_all'} for r in results]
    with open(results_path, 'wb') as f:
        pickle.dump(save_results, f)
    print(f"Results saved to {results_path}")

    # === 打印结果表格 ===
    print("\n" + "=" * 70)
    print(f"{'λ₁':>6} {'λ₂':>6} {'f₁ (Distance)':>16} {'f₂ (Waiting)':>16}")
    print("-" * 70)
    for r in results:
        print(f"{r['lambda'][0]:>6.1f} {r['lambda'][1]:>6.1f} "
              f"{r['f1_mean']:>12.2f} ± {r['f1_std']:<6.2f} "
              f"{r['f2_mean']:>8.2f} ± {r['f2_std']:<6.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
