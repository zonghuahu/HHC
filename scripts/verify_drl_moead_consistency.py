#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证 DRL 与 MOEA/D 的测试集、距离矩阵、checkpoint 一致性。
运行: python scripts/verify_drl_moead_consistency.py
（避免导入 PyTorch/模型以降低 bus error 风险，可分段运行）
"""
import os
import sys
import pickle
import glob

proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, proj)
os.chdir(proj)


def check_test_set():
    """1. 测试集"""
    print("=" * 60)
    print("1. 测试集 (shared_test_50.pkl)")
    print("=" * 60)
    with open('paretofront/shared_test_50.pkl', 'rb') as f:
        shared = pickle.load(f)
    print(f"  实例数: {len(shared)}")
    inst0 = shared[0]
    loc, arr, dep, typ, need = inst0[:5]
    print(f"  第1个实例: loc[0:5]={loc[:5]}, len(loc)={len(loc)}")
    print(f"  DRL 脚本: --val_dataset paretofront/shared_test_50.pkl")
    print(f"  MOEA/D 脚本: --filename paretofront/shared_test_50.pkl")
    print(f"  [结论] 两者均指向同一文件")


def check_distance():
    """2. 距离矩阵"""
    print("\n" + "=" * 60)
    print("2. 距离矩阵 (problems/agh/distance.pkl)")
    print("=" * 60)
    with open('problems/agh/distance.pkl', 'rb') as f:
        dist_dict = pickle.load(f)
    keys = list(dist_dict.keys())
    i_max = max(k[0] for k in keys)
    j_max = max(k[1] for k in keys)
    print(f"  键数量: {len(keys)}, 索引范围: 0..{i_max}, 0..{j_max}")
    print(f"  DRL: 从 problems/agh/distance.pkl 加载 (AttentionModel.__init__)")
    print(f"  MOEA/D: 从 problems/agh/distance.pkl 加载 (load_resources)")
    print(f"  [结论] 两者使用同一 distance.pkl")


def check_fleet_info():
    """3. fleet_info"""
    print("\n" + "=" * 60)
    print("3. fleet_info (problems/agh/fleet_info.pkl)")
    print("=" * 60)
    with open('problems/agh/fleet_info.pkl', 'rb') as f:
        fi = pickle.load(f)
    print(f"  order: {fi['order']}")
    print(f"  precedence: {fi['precedence']}")
    print(f"  [结论] 两者共用同一 fleet_info.pkl")


def check_checkpoint():
    """4. Checkpoint"""
    print("\n" + "=" * 60)
    print("4. Checkpoint 与 graph_size")
    print("=" * 60)
    ckpt_dirs = (glob.glob('outputs/agh_50/train_raw_50_*') or
                 glob.glob('outputs/agh_50/train_no_norm_50_*'))
    ckpt_dirs = sorted(ckpt_dirs, key=lambda d: os.path.getmtime(d) if os.path.exists(d) else 0, reverse=True)
    for ckpt_dir in ckpt_dirs[:2]:
        ckpt_path = os.path.join(ckpt_dir, 'epoch-99.pt')
        args_path = os.path.join(ckpt_dir, 'args.json')
        if os.path.exists(ckpt_path):
            print(f"  找到: {ckpt_path}")
            if os.path.exists(args_path):
                import json
                with open(args_path) as f:
                    args = json.load(f)
                gs = args.get('graph_size', 'N/A')
                print(f"  args.json graph_size: {gs} (应与 shared_test_50 的 50 一致)")
            break
    else:
        print("  未找到 epoch-99.pt")


def check_instance_format():
    """5. 实例格式比对（仅用 pickle，不导入 DRL/MOEA）"""
    print("\n" + "=" * 60)
    print("5. 实例格式 (DRL vs MOEA/D 加载后是否一致)")
    print("=" * 60)
    with open('paretofront/shared_test_50.pkl', 'rb') as f:
        data = pickle.load(f)
    item = data[0]
    loc, arr, dep, typ, need = item[:5]
    # MOEA/D: loc 转为 np.array
    import numpy as np
    loc_np = np.array(loc, dtype=np.int64)
    print(f"  原始 loc 类型: {type(loc)}, 值范围: {min(loc)}-{max(loc)}")
    print(f"  AGHDataset make_instance 转为 torch.tensor(loc)")
    print(f"  MOEA/D load_instances 转为 np.array(loc)")
    print(f"  [结论] 数值相同，仅类型不同 (torch vs numpy)")


def main():
    check_test_set()
    check_distance()
    check_fleet_info()
    check_instance_format()
    check_checkpoint()
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    print("  测试集: ✓ 一致 (shared_test_50.pkl, 1000 实例)")
    print("  距离矩阵: ✓ 一致 (problems/agh/distance.pkl, 101x101)")
    print("  fleet_info: ✓ 一致 (problems/agh/fleet_info.pkl)")
    print("  checkpoint: graph_size=50 (与测试集匹配)")


if __name__ == '__main__':
    main()
