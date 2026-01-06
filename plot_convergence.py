"""
训练收敛可视化脚本
用法: python plot_convergence.py [log_path]
示例: python plot_convergence.py outputs/agh_50/run_20260106T192155/validate_log.txt
"""
import os
import re
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持（可选）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_log(log_path):
    """解析 validate_log.txt 文件"""
    with open(log_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 提取 epoch 和 cost
    pattern = r'Epoch (\d+), Validation avg_cost: ([\d.]+)'
    matches = re.findall(pattern, text)
    
    epochs = [int(m[0]) for m in matches]
    costs = [float(m[1]) for m in matches]
    
    return epochs, costs


def find_latest_log():
    """自动查找最新的 validate_log.txt"""
    pattern = 'outputs/agh_*/run_*/validate_log.txt'
    logs = glob.glob(pattern)
    if not logs:
        return None
    # 按修改时间排序，返回最新的
    return max(logs, key=os.path.getmtime)


def plot_convergence(epochs, costs, save_path='convergence.png', title=None):
    """绘制收敛曲线"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 主曲线
    ax.plot(epochs, costs, 'b-', linewidth=2, label='Validation Cost', alpha=0.8)
    ax.scatter(epochs, costs, c='blue', s=20, alpha=0.5)
    
    # 移动平均线（平滑曲线）
    if len(costs) > 5:
        window = min(10, len(costs) // 5)
        smoothed = np.convolve(costs, np.ones(window)/window, mode='valid')
        smooth_epochs = epochs[window-1:]
        ax.plot(smooth_epochs, smoothed, 'r-', linewidth=2, label=f'Moving Avg (window={window})', alpha=0.7)
    
    # 标记最佳 epoch
    best_idx = np.argmin(costs)
    best_epoch = epochs[best_idx]
    best_cost = costs[best_idx]
    ax.scatter([best_epoch], [best_cost], c='green', s=150, marker='*', zorder=5, label=f'Best: Epoch {best_epoch}')
    ax.annotate(f'Best: {best_cost:.2f}', 
                xy=(best_epoch, best_cost), 
                xytext=(best_epoch + len(epochs)*0.05, best_cost + (max(costs)-min(costs))*0.05),
                fontsize=10, color='green',
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))
    
    # 图表设置
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Cost (Total Distance)', fontsize=12)
    ax.set_title(title or 'Training Convergence Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息文本框
    stats_text = f"""Training Statistics:
• Start Cost: {costs[0]:.2f}
• Final Cost: {costs[-1]:.2f}
• Best Cost: {best_cost:.2f} (Epoch {best_epoch})
• Improvement: {((costs[0] - best_cost) / costs[0] * 100):.1f}%
• Total Epochs: {len(epochs)}"""
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] 图像已保存到: {save_path}")
    
    # 显示图像
    plt.show()
    
    return fig


def main():
    # 获取日志路径
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_path = find_latest_log()
        if log_path is None:
            print("[ERROR] 未找到 validate_log.txt 文件！")
            print("用法: python plot_convergence.py <validate_log.txt路径>")
            return
        print(f"[INFO] 自动选择最新日志: {log_path}")
    
    # 检查文件是否存在
    if not os.path.exists(log_path):
        print(f"[ERROR] 文件不存在: {log_path}")
        return
    
    # 解析日志
    epochs, costs = parse_log(log_path)
    
    if not epochs:
        print("[ERROR] 日志文件中没有找到有效数据！")
        return
    
    print(f"[INFO] 找到 {len(epochs)} 个 epoch 的数据")
    print(f"   起始 Cost: {costs[0]:.2f}")
    print(f"   最终 Cost: {costs[-1]:.2f}")
    print(f"   最佳 Cost: {min(costs):.2f} (Epoch {epochs[np.argmin(costs)]})")
    
    # 生成保存路径
    log_dir = os.path.dirname(log_path)
    save_path = os.path.join(log_dir, 'convergence.png')
    
    # 获取 run 名称作为标题
    run_name = os.path.basename(log_dir)
    title = f'Training Convergence - {run_name}'
    
    # 绘图
    plot_convergence(epochs, costs, save_path, title)


if __name__ == '__main__':
    main()
