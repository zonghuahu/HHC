#!/bin/bash
#SBATCH -p gpu_a100
#SBATCH --gpus=1
#SBATCH -t 2:00:00
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -o /home/ltan/HHC/scripts/plot_convergence_%j.out

# 用最新 checkpoint 重新生成 cost 随 epoch 的收敛图
# 需在能正常加载 PyTorch 的环境运行（如 GPU 节点）

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hhc
cd /home/ltan/HHC

echo "=== Regenerating convergence plot with latest data ==="

# 自动选择最新 train_raw_50 或 train_no_norm_50
CKPT_50=$(ls -td outputs/agh_50/train_raw_50_* 2>/dev/null | head -1)
[ -z "$CKPT_50" ] && CKPT_50=$(ls -td outputs/agh_50/train_no_norm_50_* 2>/dev/null | head -1)
[ -z "$CKPT_50" ] && CKPT_50="outputs/agh_50/train_raw_50_20260228T011907"

echo "Using checkpoint dir: $CKPT_50"

# 全部 100 个 epoch 完整评估
python -u visualization/plot_convergence.py \
    --checkpoint_dir "$CKPT_50" \
    --exp_name convergence

echo ""
echo "Plot saved to: images/convergence/convergence.png"
