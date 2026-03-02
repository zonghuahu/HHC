#!/bin/bash
#SBATCH -p gpu_a100
#SBATCH --gpus=1
#SBATCH -t 2:00:00
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -o /home/ltan/HHC/scripts/plot_convergence_200ep_%j.out

# 生成 n=50 和 n=100 的 200 epoch 多 λ 收敛图，保存到 images/convergence/50_100_200ep.png

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hhc
cd /home/ltan/HHC

echo "=== Generating 200ep multi-λ convergence plot (n=50 + n=100) ==="

CKPT_50=$(ls -td outputs/agh_50/train_raw_50_200ep_* 2>/dev/null | head -1)
[ -z "$CKPT_50" ] && CKPT_50="outputs/agh_50/train_raw_50_200ep_20260228T162007"

CKPT_100=$(ls -td outputs/agh_100/train_raw_100_200ep_* 2>/dev/null | head -1)
[ -z "$CKPT_100" ] && CKPT_100="outputs/agh_100/train_raw_100_200ep_20260228T162008"

echo "n=50:  $CKPT_50"
echo "n=100: $CKPT_100"

# sample_epochs=5 每 5 个 epoch 评估一次，共约 40 个点，加快速度
if [ -d "$CKPT_100" ]; then
    python -u visualization/plot_convergence.py \
        --save images/convergence/50_100_200ep.png \
        --checkpoint_dir_50 "$CKPT_50" \
        --checkpoint_dir_100 "$CKPT_100" \
        --sample_epochs 5
    echo ""
    echo "Plot saved to: images/convergence/50_100_200ep.png"
else
    echo "[WARN] n=100 200ep dir not found, generating n=50 only"
    python -u visualization/plot_convergence.py \
        --save images/convergence/50_200ep.png \
        --checkpoint_dir "$CKPT_50" \
        --sample_epochs 5
    echo ""
    echo "Plot saved to: images/convergence/50_200ep.png"
fi
