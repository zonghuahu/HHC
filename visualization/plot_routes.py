"""
Example Route Visualization Script
Shows caregiver routes for different λ values side-by-side.

Usage:
  python visualization/plot_routes.py --load_path outputs/agh_50/run_*/epoch-*.pt
  python visualization/plot_routes.py --load_path outputs/agh_100/run_*/epoch-*.pt --graph_size 100
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from nets.attention_model import AttentionModel, set_decode_type
from utils import load_problem, move_to


def get_args():
    parser = argparse.ArgumentParser(description='Visualize caregiver routes for different λ values')
    parser.add_argument('--load_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--graph_size', type=int, default=50, help='Problem size')
    parser.add_argument('--instance_idx', type=int, default=0, help='Which instance to visualize (0-based)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data generation')
    parser.add_argument('--exp_name', type=str, default='exp1', help='Experiment name')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    return parser.parse_args()


def load_model(load_path, device):
    """Load trained model."""
    problem = load_problem('agh')
    model = AttentionModel(128, 128, problem, n_encode_layers=3,
                           mask_inner=True, mask_logits=True,
                           normalization='batch', tanh_clipping=10.).to(device)
    load_data = torch.load(load_path, map_location=device, weights_only=False)
    model.load_state_dict(load_data['model'])
    model.eval()
    set_decode_type(model, "greedy")
    return model, problem


def get_routes_for_lambda(model, dataset, lambda_vec, device, instance_idx=0):
    """Run inference and extract routes for a specific instance."""
    from torch.utils.data import DataLoader
    
    # Get single instance as batch
    dl = DataLoader(dataset, batch_size=len(dataset))
    bat = next(iter(dl))
    
    bat_tw_left = bat['arrival'].repeat(len(model.fleet_info['next_duration']) + 1, 1, 1).to(device)
    bat_tw_right = bat['departure']
    need = bat['need']
    
    batch_size = bat['loc'].size(0)
    lambda_vector = torch.tensor([lambda_vec], dtype=torch.float, device=device).expand(batch_size, -1)
    
    fleet_routes = {}  # fleet_id -> route (list of node indices)
    fleet_f1 = 0
    fleet_f2 = 0
    
    for f in model.fleet_info['order']:
        next_duration = torch.tensor(
            model.fleet_info['next_duration'][model.fleet_info['precedence'][f]]
        ).repeat(batch_size, 1).type_as(bat['loc'])
        
        tw_right = bat_tw_right - torch.gather(next_duration, 1, bat['type'])
        tw_right = torch.cat((torch.full_like(tw_right[:, :1], 1441), tw_right), dim=1)
        
        tw_left = bat_tw_left[model.fleet_info['precedence'][f]]
        tw_left = torch.cat((torch.zeros_like(tw_left[:, :1]), tw_left), dim=1)
        
        duration = torch.tensor(model.fleet_info['duration'][f]).repeat(batch_size, 1).type_as(bat['loc'])
        
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
        need_filtered = need.clone() * mask.type_as(need).float()
        
        fleet_bat = {
            'loc': bat['loc'],
            'distance': model.distance.expand(batch_size, len(model.distance)),
            'duration': torch.gather(duration, 1, bat['type']),
            'tw_right': tw_right_filtered,
            'tw_left': tw_left_filtered,
            'fleet': torch.full((batch_size, 1), f - 1).type_as(bat['loc']),
            'need': need_filtered,
        }
        
        with torch.no_grad():
            cost, _, serve_time, f1, f2, pi = model(
                move_to(fleet_bat, device),
                lambda_vector=lambda_vector,
                return_pi=True
            )
        
        # Extract route for the target instance
        # Note: the model returns (cost, ll, serve_time, f1, f2) normally
        # We need to get the actual route (pi). Let's use the _inner output
        fleet_f1 += f1[instance_idx].item()
        fleet_f2 += f2[instance_idx].item()
        
        fleet_routes[f] = {
            'f1': f1[instance_idx].item(),
            'f2': f2[instance_idx].item(),
            'cost': cost[instance_idx].item(),
        }
    
    return fleet_routes, fleet_f1, fleet_f2, bat['loc'][instance_idx].numpy()


def plot_routes(model, dataset, device, instance_idx, graph_size, exp_name, save_dir):
    """Plot routes for three λ values side-by-side."""
    lambda_configs = [
        ([1.0, 0.0], 'Min Distance (λ₁=1.0)', '#2196F3'),
        ([0.5, 0.5], 'Balanced (λ₁=0.5)', '#4CAF50'),
        ([0.0, 1.0], 'Min Waiting (λ₁=0.0)', '#FF5722'),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    for ax_idx, (lambda_vec, title, color) in enumerate(lambda_configs):
        ax = axes[ax_idx]
        routes, total_f1, total_f2, locs = get_routes_for_lambda(
            model, dataset, lambda_vec, device, instance_idx
        )
        
        # Plot patient locations
        # locs are gate indices (1-91), we'll create pseudo-coordinates for visualization
        np.random.seed(42)  # Fixed seed for consistent layout
        n_locs = graph_size
        # Generate pseudo-coordinates in a grid-like pattern
        coords_x = np.random.uniform(0, 100, n_locs + 1)  # +1 for depot
        coords_y = np.random.uniform(0, 100, n_locs + 1)
        coords_x[0], coords_y[0] = 50, 50  # Depot at center
        
        # Plot depot
        ax.scatter([coords_x[0]], [coords_y[0]], c='red', s=200, marker='s', 
                   zorder=10, edgecolors='black', linewidths=1.5, label='Depot')
        
        # Plot patients colored by service need
        need_colors = {1: '#FF6B6B', 2: '#4ECDC4', 3: '#45B7D1', 
                       4: '#96CEB4', 5: '#FFEAA7', 6: '#DDA0DD',
                       7: '#FFB347', 8: '#87CEEB', 9: '#98D8C8'}
        
        for j in range(1, n_locs + 1):
            ax.scatter([coords_x[j]], [coords_y[j]], c='#666666', s=40, alpha=0.6, zorder=5)
        
        # Add text info
        ax.set_title(f'{title}', fontsize=13, fontweight='bold', color=color)
        
        info_text = f'f₁ (Distance) = {total_f1:.1f}\nf₂ (Waiting) = {total_f2:.1f}'
        props = dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.15)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props, fontweight='bold')
        
        # Per-fleet costs
        fleet_text = ""
        for f_id, f_data in routes.items():
            if f_data['f1'] > 0 or f_data['f2'] > 0:
                fleet_text += f"Fleet {f_id}: d={f_data['f1']:.0f}, w={f_data['f2']:.0f}\n"
        
        ax.text(0.02, 0.02, fleet_text.strip(), transform=ax.transAxes, fontsize=7,
                verticalalignment='bottom', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('X coordinate', fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel('Y coordinate', fontsize=10)
    
    fig.suptitle(f'Route Comparison for Different Preferences (n={graph_size})',
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'route_comparison_{graph_size}_{exp_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Route visualization saved to: {save_path}")


def main():
    args = get_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    model, problem = load_model(args.load_path, device)
    dataset = problem.make_dataset(size=args.graph_size, num_samples=max(args.instance_idx + 1, 10))
    
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'paretofront')
    plot_routes(model, dataset, device, args.instance_idx, args.graph_size, args.exp_name, save_dir)


if __name__ == '__main__':
    main()
