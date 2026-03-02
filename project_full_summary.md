# HHCRSP 多目标深度强化学习项目 — 完整技术总结

**生成日期**: 2026-03-02  
**工作目录**: `/home/ltan/HHC`  
**分支**: `Zonghua_Hu` (ahead 4 commits from origin)  
**目的**: 本文档记录项目全部技术细节，作为代码回滚前的完整参考

---

## 目录

1. [项目概述](#1-项目概述)
2. [问题定义与数学模型](#2-问题定义与数学模型)
3. [数据结构与静态资源](#3-数据结构与静态资源)
4. [DRL 方法：模型架构](#4-drl-方法模型架构)
5. [DRL 方法：训练流程](#5-drl-方法训练流程)
6. [DRL 方法：推理流程](#6-drl-方法推理流程)
7. [MOEA/D 基线方法](#7-moead-基线方法)
8. [对比实验设计](#8-对比实验设计)
9. [实验结果与分析](#9-实验结果与分析)
10. [可视化系统](#10-可视化系统)
11. [SLURM 集群配置](#11-slurm-集群配置)
12. [完整文件清单](#12-完整文件清单)
13. [Git 提交历史](#13-git-提交历史)
14. [训练运行记录](#14-训练运行记录)
15. [已知问题与改进方向](#15-已知问题与改进方向)
16. [复现指南](#16-复现指南)

---

## 1. 项目概述

### 1.1 研究主题

硕士论文项目：**多目标深度强化学习在家庭护理路径调度问题（HHCRSP）上的应用**。

- **DRL 方法**：基于 Attention Model (Kool et al., 2019) + WE-Add（Weight Embedding Addition）权重嵌入
- **Baseline 方法**：手写 MOEA/D (moead.py) + 可选 pymoo 封装 (hhc_problem_pymoo.py)
- **两个优化目标**：f₁（总行驶距离）和 f₂（总患者等待时间）
- **代码基础**：基于机场地勤（AGH）问题改造为家庭护理（HHC）问题

### 1.2 关键术语映射

代码中沿用了 AGH 术语，实际对应关系如下：

| AGH 术语 | HHC 含义 | 说明 |
|----------|---------|------|
| gate | 患者住址位置 | 1–100，共 100 个固定位置 |
| fleet | 护理人员/团队类型 | 6 种，编号 1–6 |
| depot (node 0) | 护理团队出发基地 | 固定起点 |
| need 1–6 | 单需求患者 | 只需一个 fleet 服务 |
| need 7/8/9 | 组合需求患者 | 需要两个 fleet 按顺序服务 |

### 1.3 问题规模

| 参数 | 值 |
|------|-----|
| NODE_SIZE | 101（100 个位置 + 1 个 depot） |
| distance.pkl | 101×101 = 10201 条目 |
| 患者数 n | 50 或 100 |
| Fleet 数量 | 6 个，按固定顺序 [1,2,3,4,5,6] 处理 |
| 服务需求分布 | 1–6 各 11.6%，7–9 各 10%（约 70%/30% 单需求/组合需求） |

---

## 2. 问题定义与数学模型

### 2.1 优化目标

$$\min \quad (f_1, f_2) = (\text{总行驶距离}, \text{总患者等待时间})$$

其中：
- f₁ = Σ（所有 fleet 路径距离之和），包括往返 depot
- f₂ = Σ max(0, service_start_time_i - tw_left_i)，即每个患者从最早可服务时间到实际服务开始的等待

### 2.2 标量化成本函数

```
cost = λ₁ · f₁ + λ₂ · f₂    （原始加权和，无归一化）
```

- 训练时：λ₁ ~ Uniform(0, 1)，λ₂ = 1 - λ₁，每个样本独立采样
- 验证时：固定 λ = (0.5, 0.5)
- 推理时：使用 101 个均匀权重向量 λ₁ ∈ {0.00, 0.01, ..., 1.00}

### 2.3 约束条件

#### 基本约束
1. **时间窗约束**：`service_start + duration ≤ tw_right`
2. **Fleet-Need 匹配**：只有特定 fleet 可服务特定 need

#### Fleet-Need 匹配表

| Fleet | 可服务的 Need | 说明 |
|-------|-------------|------|
| 1 | 1, 9 | 单需求 1 + 组合需求 9 的第一阶段 |
| 2 | 2, 7 | 单需求 2 + 组合需求 7 的第一阶段 |
| 3 | 3, 7 | 单需求 3 + 组合需求 7 的第二阶段 |
| 4 | 4, 8 | 单需求 4 + 组合需求 8 的第一阶段 |
| 5 | 5, 8 | 单需求 5 + 组合需求 8 的第二阶段 |
| 6 | 6, 9 | 单需求 6 + 组合需求 9 的第二阶段 |

#### 组合需求（Combo）约束

仅对 combo 患者（need=7/8/9）生效，不对 single 患者（need=1–6）生效：

```python
COMBO_NEED = {3: 7, 5: 8, 6: 9}
LATE_TOLERANCE = {3: 30.0, 5: 30.0, 6: 0.0}
```

| Fleet | Combo Need | 约束 | 含义 |
|-------|-----------|------|------|
| 3 | need=7 | arrival ≤ tw_left + 30 min | Fleet 3 服务 combo 患者不得迟到超过 30 分钟 |
| 5 | need=8 | arrival ≤ tw_left + 30 min | Fleet 5 服务 combo 患者不得迟到超过 30 分钟 |
| 6 | need=9 | arrival ≤ tw_left + 0 min | Fleet 6 服务 combo 患者零容忍迟到 |

#### 顺序优先约束

- Fleet 处理顺序固定：[1, 2, 3, 4, 5, 6]
- 组合需求患者：第一个 fleet 完成后，10 分钟间隔传递到第二个 fleet（fleet 1 为 0 分钟间隔）
- 通过 `fleet_info['precedence']` 和 `fleet_info['next_duration']` 控制 tw_left 传播

#### Fleet Info 结构

```python
fleet_info = {
    'order': [1, 2, 3, 4, 5, 6],
    'precedence': {1: 2, 2: 0, 3: 1, 4: 0, 5: 1, 6: 3},
    'duration': {
        1: [30.0, 30.0, 30.0],  # 按 type 0/1/2 的服务时长
        2: [30.0, 31.0, 31.0],
        3: [34.0, 33.0, 32.0],
        4: [29.0, 31.0, 30.0],
        5: [32.0, 31.0, 30.0],
        6: [30.0, 30.0, 30.0]
    },
    'next_duration': {
        3: [0.0, 0.0, 0.0],
        2: [0.0, 0.0, 0.0],
        1: [0.0, 0.0, 0.0],
        0: [60.0, 60.0, 60.0]
    }
}
```

- `precedence[f]` 表示 fleet f 读取哪个 stage 的 tw_left
- `next_duration[stage]` 用于收紧 tw_right（`tw_right = departure - next_duration[type]`）
- 车辆速度：SPEED = 80.0（距离单位/分钟）
- 初始空闲时间：INIT_FREE_TIME = -60（提前 60 分钟开始）

---

## 3. 数据结构与静态资源

### 3.1 静态资源文件

| 文件 | 路径 | 内容 |
|------|------|------|
| 距离矩阵 | `problems/agh/distance.pkl` | 101×101 字典 {(i,j): dist} |
| 车队信息 | `problems/agh/fleet_info.pkl` | order, precedence, duration, next_duration |
| 到达概率 | `problems/agh/arrival_prob.npy` | 10 个小时（10:00–19:00）的到达概率分布 |
| 节点坐标 | `problems/agh/coordinates.pkl` | 101 个节点的坐标 |

### 3.2 实例数据格式

每个实例是一个 5 元组：`(loc, arrival, departure, type_, need)`

| 字段 | 类型 | 范围 | 说明 |
|------|------|------|------|
| loc | int[n] | 1–100 | 患者位置索引 |
| arrival | float[n] | 600–1199 | 到达时间（分钟，10:00–19:59） |
| departure | float[n] | arrival + 120 | 固定 120 分钟窗口 |
| type_ | int[n] | 0, 1, 2 | 3 种类型，影响服务时长 |
| need | int[n] | 1–9 | 服务需求类型 |

### 3.3 数据生成

**训练数据**：每个 epoch 在 `AGHDataset.__init__` 中实时随机生成，不保存到文件。

**共享测试数据**：通过 `generate_test_data.py` 一次性生成并保存：

```bash
python generate_test_data.py --n_instances 1000 --seed 1234 --output_dir paretofront
# 输出: paretofront/shared_test_50.pkl, paretofront/shared_test_100.pkl
```

---

## 4. DRL 方法：模型架构

### 4.1 整体架构

**Attention Model** = Transformer Encoder + Autoregressive Decoder

```
输入特征 → 初始嵌入 → [WE-Add: + λ 嵌入] → N 层 Graph Attention Encoder → 节点嵌入
                                                                           ↓
                              ← 自回归解码（逐步选择节点）← 上下文查询 + 多头注意力
```

### 4.2 模型参数

| 参数 | 值 | 说明 |
|------|-----|------|
| embedding_dim | 128 | 节点嵌入维度 |
| hidden_dim | 128 | 隐藏层维度 |
| n_encode_layers | 3 | Encoder 层数 |
| n_heads | 8 | 多头注意力头数 |
| tanh_clipping | 10.0 | logits 裁剪范围 |
| normalization | batch | 批归一化 |
| feed_forward_hidden | 512 | Encoder 前馈层隐藏维度 |

### 4.3 输入特征嵌入（AGH）

1. **位置嵌入**：`loc_embedding = nn.Embedding(101, 128)` — 100 个位置 + 1 个 depot
2. **时间窗特征嵌入**：`init_embed = nn.Linear(2, 128)` — 输入 (tw_left/1440, tw_right/1440)
3. **车队嵌入**：`fleets_embedding = nn.Embedding(6, 128)` — 6 个 fleet
4. **初始嵌入** = 位置嵌入 + 时间窗特征嵌入（逐元素相加）
5. **无效节点屏蔽**：tw_right ≤ 0 的节点嵌入置零

### 4.4 WE-Add 权重嵌入机制

在 `GraphAttentionEncoder` 中：

```python
self.W_lambda = nn.Linear(2, 128)  # λ = [λ₁, λ₂] → 128 维嵌入

# forward 中：
h_lambda = self.W_lambda(lambda_val)       # [batch_size, 128]
h = h + h_lambda.unsqueeze(1)              # broadcast 到所有节点
```

- 将 λ 向量映射为嵌入，加到所有节点的初始嵌入上
- 使同一模型能根据不同 λ 生成不同路径策略
- 仅需训练一个模型，推理时切换 λ 即可获得 Pareto 前沿

### 4.5 Encoder：GraphAttentionEncoder

每层结构（共 3 层）：
```
SkipConnection(MultiHeadAttention) → BatchNorm → SkipConnection(FFN) → BatchNorm
```

其中：
- `MultiHeadAttention`：8 头自注意力，key_dim = val_dim = 128/8 = 16
- `FFN`：Linear(128→512) → ReLU → Linear(512→128)

### 4.6 Decoder：自回归节点选择

每步解码过程：

1. **上下文构造**：当前节点嵌入 + 当前空闲时间(cur_free_time/1440)
2. **查询计算**：`query = 全局上下文投影 + 步骤上下文投影 + 车队嵌入`
3. **多头注意力**：query 与所有节点的 key/value 做注意力计算
4. **logits 计算**：注意力输出与 logit_key 点积 → tanh 裁剪 → mask 不可行节点
5. **节点选择**：
   - 训练：`sampling`（多项式采样）
   - 推理：`greedy`（取最大概率）
6. **状态更新**：更新 visited mask、cur_free_time、serve_time、tour

### 4.7 Masking 逻辑（state_agh.py `get_mask()`）

对每个 fleet_type 分别处理：

- **Fleet 1, 2, 4**（无 combo 约束）：
  ```
  mask = (arrival + duration > tw_right)
  ```

- **Fleet 3, 5**（有 30 分钟 combo 约束）：
  - scenario1（针对非 combo need）：`arrival + duration > tw_right`
  - scenario2（针对 combo need=7/8）：`arrival > tw_left + 30` 或 `adjusted_arrival + duration > tw_right`
  - `mask = scenario1 AND scenario2`（两种场景的并集取反）

- **Fleet 6**（零容忍 combo 约束）：
  - scenario1（针对非 combo need=9）：`arrival + duration > tw_right`
  - scenario2（针对 combo need=9→6）：`arrival > tw_left` 或 `adjusted_arrival + duration > tw_right`
  - `mask = scenario1 AND scenario2`

- **Depot 约束**：如果刚访问 depot 且仍有可访问节点，禁止再次访问 depot

### 4.8 Cost 计算（problem_agh.py `get_costs()`）

```python
def get_costs(dataset, pi, return_components=False):
    # 逐步模拟路径执行
    for i in range(pi.size(1)):
        arrival_time = cur_time + travel_time[i]
        service_start = max(arrival_time, tw_left[node])
        waiting_i = max(0, service_start - tw_left[node])  # 患者等待时间
        total_waiting += waiting_i
        cur_time = (service_start + duration[node]) * is_not_depot - 60 * is_depot
    
    total_distance = sum(all_segment_distances)
    
    if return_components:
        return total_distance, total_waiting, None  # f₁, f₂
    return total_distance, None  # 向后兼容
```

---

## 5. DRL 方法：训练流程

### 5.1 训练参数

| 参数 | n=50 | n=100 | 说明 |
|------|------|-------|------|
| n_epochs | 100（或 200） | 100（或 200） | 训练轮数 |
| epoch_size | 12800 | 12800 | 每轮训练实例数 |
| batch_size | 64 | 64 | 批次大小 |
| batches/epoch | 200 | 200 | 12800/64 |
| optimizer | Adam | Adam | |
| lr_model | 1e-4 | 1e-4 | Actor 学习率 |
| lr_critic | 1e-4 | 1e-4 | Critic 学习率（当前未使用） |
| lr_decay | 1.0 | 1.0 | 无衰减 |
| max_grad_norm | 1.0 | 1.0 | 梯度裁剪 |
| baseline | rollout | rollout | 基线类型 |
| bl_alpha | 0.05 | 0.05 | t 检验显著性水平 |
| bl_warmup_epochs | 0 | 0 | 无预热 |
| seed | 123456 | 123456 | 训练随机种子 |
| eval_batch_size | 1000 | 1000 | 验证批次大小 |
| val_size | 1000 | 1000 | 验证集大小 |
| checkpoint_epochs | 1 | 1 | 每轮保存 |

### 5.2 训练循环详解

```
for epoch in range(n_epochs):
    # 1. 生成训练数据（每轮重新生成 12800 个实例）
    training_dataset = baseline.wrap_dataset(problem.make_dataset(...))
    # ↑ wrap_dataset 会对数据执行一次 rollout，获取基线值 bl_vals
    
    # 2. 遍历 200 个 batch（batch_size=64）
    for batch in training_dataloader:
        train_batch_agh(...)
    
    # 3. 保存 checkpoint
    torch.save(checkpoint, 'epoch-{epoch}.pt')
    
    # 4. 验证（用固定 λ=(0.5,0.5) 做 greedy 评估）
    avg_cost = validate(model, val_dataset, opts)
    
    # 5. Rollout baseline 回调（t 检验决定是否更新）
    baseline.epoch_callback(model, epoch)
    
    # 6. 学习率调度（lr_decay=1.0 实际无衰减）
    lr_scheduler.step()
    
    # 7. 清理 GPU 缓存
    torch.cuda.empty_cache()
```

### 5.3 单批次训练流程（train_batch_agh）

```
1. 解包数据和基线值
2. 采样随机 λ: λ₁ ~ Uniform(0,1), λ₂ = 1 - λ₁  [batch_size, 2]

3. Policy forward pass（6 个 fleet 顺序解码）:
   for f in [1, 2, 3, 4, 5, 6]:
       a. 构造 fleet 输入（mask 无关患者）
       b. 前向传播: fleet_cost, log_likelihood, serve_time, f1, f2 = model(fleet_bat, lambda_vector=λ)
       c. 累加 f1, f2, log_likelihood
       d. 传播 tw_left 到下一个 stage

4. Baseline with SAME λ:
   bl_cost_list, bl_f1_list, bl_f2_list = baseline.eval_agh(x, fleet_info, distance, λ, opts)

5. 标量化成本:
   cost_policy  = λ₁ · Σf1_policy  + λ₂ · Σf2_policy
   cost_baseline = λ₁ · Σf1_baseline + λ₂ · Σf2_baseline

6. REINFORCE:
   advantage = cost_policy - cost_baseline
   loss = mean(advantage * Σlog_likelihood)

7. 反向传播 + 梯度裁剪(max_norm=1.0) + 参数更新
```

### 5.4 Rollout Baseline 机制

- 维护一个基线模型（训练模型的深拷贝）
- 每个 epoch 结束后，用候选模型（当前训练模型）在验证集上评估
- 执行单侧 t 检验（scipy ttest_rel），若 p < 0.05 则用候选模型替换基线模型
- wrap_dataset 时，基线模型以 greedy 模式对训练数据执行一次 rollout，得到 bl_vals

### 5.5 Checkpoint 内容

```python
{
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'rng_state': torch.get_rng_state(),
    'cuda_rng_state': torch.cuda.get_rng_state_all(),
    'baseline': baseline.state_dict(),  # 包含基线模型和数据集
}
```

---

## 6. DRL 方法：推理流程

### 6.1 Pareto 推理脚本

`visualization/pareto_inference.py`：

```bash
python -u visualization/pareto_inference.py \
    --load_path outputs/agh_50/train_raw_50_200ep_*/epoch-99.pt \
    --graph_size 50 \
    --val_size 1000 \
    --val_dataset paretofront/shared_test_50.pkl \
    --seed 1234 \
    --exp_name raw_101w \
    --output_dir paretofront
```

### 6.2 推理流程

```
1. 加载模型 checkpoint
2. 加载共享测试数据（shared_test_*.pkl）
3. 对 101 个 λ 权重向量（λ₁ = 0.00, 0.01, ..., 1.00）:
   a. 设定 lambda_vector = [λ₁, λ₂]
   b. 6 个 fleet 顺序 greedy 解码
   c. 收集每个实例的 (f₁, f₂)
   d. 计算均值和标准差
4. 绘制 Pareto 前沿图
5. 保存 .pkl 和 .txt 结果
```

### 6.3 输出文件

| 文件 | 内容 |
|------|------|
| `paretofront/pareto_results_{n}_raw_101w.pkl` | DRL 结果数据 |
| `paretofront/pareto_results_{n}_raw_101w.txt` | 文本格式结果 |
| `images/raw_101w/pareto_front_{n}_raw_101w.png` | Pareto 前沿图 |

---

## 7. MOEA/D 基线方法

### 7.1 算法框架

MOEA/D（Multi-Objective Evolutionary Algorithm based on Decomposition）使用 Tchebycheff 分解。

**Tchebycheff 标量化**：
```
g(f₁, f₂ | w, z*) = max(w₁ · |f₁ - z*₁|, w₂ · |f₂ - z*₂|)
```
其中 z* = (min f₁, min f₂) 为动态更新的理想点。

### 7.2 算法参数

| 参数 | 值 | 说明 |
|------|-----|------|
| n_weights | 101 | 权重向量数（λ₁ = 0.00, 0.01, ..., 1.00） |
| T | 3 | 邻域大小 |
| n_gen | 500 | 进化代数 |
| mutation_rate | 0.3 | 变异概率 |
| seed | 1234 | 随机种子 |
| n_instances | 1000 | 测试实例数 |

### 7.3 初始解构造（3 种混合）

1. **Sorted Nearest Neighbor**：按 tw_left 排序，优先选最早+最近，通常质量最好
2. **Regular Nearest Neighbor**：经典最近邻，纯距离导向
3. **Random Construction**：随机打乱后贪心插入

种群初始化：第 0 个用 sorted-NN，偶数索引用 regular-NN，奇数索引用 random。

### 7.4 遗传算子

- **交叉**：Order Crossover (OX) — 保持部分顺序的交叉
- **变异**（各 1/3 概率）：
  - `mutate_swap`：交换两个随机患者位置
  - `mutate_relocate`：将随机患者移到随机位置
  - `mutate_reverse_segment`：反转随机片段（类似 2-opt）

### 7.5 进化流程

```
对每个测试实例独立优化:
  1. 初始化 n_weights 个解
  2. 计算初始目标值和理想点 z*
  3. 迭代 500 代:
     对每个子问题 i:
       a. 从邻域随机选两个父代
       b. OX 交叉 → 两个子代，随机选一个
       c. 对选中子代应用变异（概率 0.3）
       d. 评估子代 (f₁, f₂)
       e. 更新 z*
       f. 对邻域中每个解 j，用 Tchebycheff 判断是否替换
```

### 7.6 评估函数对齐

`evaluate_solution()` 严格复制 DRL 的顺序评估逻辑：

- 6 个 fleet 按 [1,2,3,4,5,6] 顺序处理
- tw_left 通过 `fleet_info['precedence']` 从对应 stage 读取
- tw_right 通过 `next_duration` 收紧
- fleet 1 完成后 0 分钟间隔传播，其他 fleet 10 分钟间隔
- 回仓时 cur_time 重置为 -60
- `is_visit_feasible` 中的 combo 约束与 DRL `state_agh.py` 完全一致

### 7.7 运行命令

```bash
python -u moead.py \
    --graph_size 50 \
    --filename paretofront/shared_test_50.pkl \
    --n_instances 1000 \
    --n_weights 101 \
    --n_gen 500 \
    --seed 1234 \
    --output_dir paretofront
```

### 7.8 运行时间

| 规模 | 实例数 | 权重数 | 代数 | 运行时间 |
|------|--------|--------|------|---------|
| n=50 | 1000 | 11 | 500 | ~40 分钟 |
| n=100 | 1000 | 11 | 500 | ~67 分钟 |
| n=100 | 1000 | 101 | 500 | ~10.4 小时 |

---

## 8. 对比实验设计

### 8.1 公平性保障清单

| 条件 | DRL | MOEA/D | 对齐状态 |
|------|-----|--------|---------|
| 测试数据 | `paretofront/shared_test_*.pkl` | 同一文件 | ✅ |
| 实例数 | 1000 | 1000 | ✅ |
| λ 权重向量 | 101 个 (0.00–1.00) | 101 个 (0.00–1.00) | ✅ |
| 随机种子 | 1234 | 1234 | ✅ |
| Fleet 处理顺序 | [1,2,3,4,5,6] | 同 | ✅ |
| tw_left 传播规则 | fleet 1: 0 gap; others: +10 | 同 | ✅ |
| tw_right 收紧 | next_duration | 同 | ✅ |
| 距离矩阵 | `problems/agh/distance.pkl` | 同 | ✅ |
| Combo 约束 | `COMBO_NEED={3:7, 5:8, 6:9}` | 同 | ✅ |
| 回仓重置 | cur_time = -60 | 同 | ✅ |

### 8.2 重要修复记录

**约束对齐修复**：MOEA/D 的 `is_visit_feasible` 中，combo 患者迟到约束曾错误地施加给 single-need 患者。已修正为仅对 combo-need 患者生效，与 DRL `state_agh.py` 的 masking 逻辑一致。

---

## 9. 实验结果与分析

### 9.1 最新 DRL 结果（n=50, 100 epochs, 101 权重, 1000 实例）

来源：`paretofront/pareto_results_50_raw_101w.txt`  
Checkpoint：`outputs/agh_50/train_raw_50_200ep_20260228T162007/epoch-99.pt`

| λ₁ | f₁ (距离) | f₂ (等待) |
|-----|----------|----------|
| 0.00 | 4776 ± 373 | 1.0 ± 5.6 |
| 0.10 | 4318 ± 369 | 15.2 ± 26.5 |
| 0.20 | 3621 ± 329 | 73.8 ± 62.1 |
| 0.30 | 3175 ± 267 | 157.4 ± 88.5 |
| 0.40 | 3030 ± 246 | 204.8 ± 97.2 |
| 0.50 | 2984 ± 237 | 223.7 ± 99.1 |
| 0.60 | 2970 ± 236 | 233.0 ± 102.6 |
| 0.70 | 2969 ± 234 | 238.8 ± 104.8 |
| 0.80 | 2971 ± 233 | 239.5 ± 102.4 |
| 0.90 | 2969 ± 234 | 239.8 ± 104.6 |
| 1.00 | 2972 ± 234 | 240.7 ± 106.7 |

### 9.2 最新 DRL 结果（n=100, 100 epochs, 101 权重, 1000 实例）

来源：`paretofront/pareto_results_100_raw_101w.txt`  
Checkpoint：`outputs/agh_100/train_raw_100_20260228T011907/epoch-85.pt`

| λ₁ | f₁ (距离) | f₂ (等待) |
|-----|----------|----------|
| 0.00 | 9764 ± 525 | 2.9 ± 13.4 |
| 0.10 | 9435 ± 557 | 25.7 ± 39.3 |
| 0.20 | 8641 ± 583 | 107.4 ± 81.1 |
| 0.30 | 7706 ± 560 | 298.3 ± 143.8 |
| 0.40 | 6892 ± 495 | 597.1 ± 200.9 |
| 0.50 | 6408 ± 415 | 891.3 ± 225.6 |
| 0.60 | 6229 ± 381 | 1116.9 ± 233.1 |
| 0.70 | 6194 ± 379 | 1225.4 ± 225.5 |
| 0.80 | 6197 ± 376 | 1262.6 ± 224.3 |
| 0.90 | 6209 ± 378 | 1291.8 ± 227.3 |
| 1.00 | 6210 ± 375 | 1288.4 ± 230.5 |

### 9.3 最新 MOEA/D 结果（n=100, 500 代, 101 权重, 1000 实例）

来源：`paretofront/moead_results_100_exp1.txt`  
运行时间：37447 秒（~10.4 小时）  
Hypervolume：739340.45 (ref: f₁=6550.07, f₂=511.76)

| λ₁ | f₁ (距离) | f₂ (等待) |
|-----|----------|----------|
| 0.00 | 5955 ± 506 | 0.03 ± 0.30 |
| 0.10 | 5390 ± 299 | 41.9 ± 16.1 |
| 0.20 | 5270 ± 284 | 71.9 ± 27.7 |
| 0.30 | 5206 ± 280 | 101.7 ± 39.4 |
| 0.40 | 5154 ± 277 | 131.4 ± 49.8 |
| 0.50 | 5119 ± 274 | 157.1 ± 60.2 |
| 0.60 | 5083 ± 275 | 190.4 ± 72.8 |
| 0.70 | 5051 ± 274 | 233.0 ± 88.6 |
| 0.80 | 5016 ± 272 | 282.2 ± 103.7 |
| 0.90 | 4979 ± 272 | 371.1 ± 128.0 |
| 1.00 | 4966 ± 274 | 446.6 ± 146.3 |

### 9.4 MOEA/D 结果（n=50, 初步版本，5 实例，5 代）

来源：`paretofront/moead_results_50_exp1.txt`  
**注意**：n=50 的 MOEA/D 当前仅有非常小规模的初步测试（5 实例、5 代），不具备统计意义。正式 1000 实例版本尚未运行。

### 9.5 结果分析

#### n=50 关键发现（基于早期 11 权重实验数据）

| 发现 | 详情 |
|------|------|
| DRL f₁ 在 λ₁≥0.3 时更优 | 距离降低 1–2% |
| MOEA/D f₂ 一致更低 | 构造启发式的时间排序偏向 |
| DRL Pareto 前沿更广 | f₁ 跨度 ~500 vs MOEA/D ~300 |

#### n=100 关键发现（101 权重正式实验）

| 发现 | 详情 |
|------|------|
| DRL f₁ 在 λ₁≥0.3 时显著更优 | 但在 λ₁ ∈ [0, 0.2] 区间 DRL f₁ 远高于 MOEA/D |
| n=100 DRL 的 f₁ 大约是 n=50 的 2 倍 | f₁: 6200–9800 vs 2970–4780 |
| n=100 DRL 的 f₂ 远大于 MOEA/D | f₂ 在 λ₁=1.0 时：DRL 1288 vs MOEA/D 447 |
| MOEA/D 的 Pareto 前沿更紧凑 | f₁ 跨度 ~1000，f₂ 跨度 ~466 |
| DRL 的 λ 响应更强 | f₁ 从 9764→6210，f₂ 从 3→1288 |

#### 速度对比

| 方法 | n=50 | n=100 |
|------|------|-------|
| DRL 训练 | ~4h (GPU) | ~12h (GPU) |
| DRL 推理 (1000 实例) | ~48s (GPU) | ~83s (GPU) |
| MOEA/D (11 权重) | ~40min (CPU) | ~67min (CPU) |
| MOEA/D (101 权重) | — | ~10.4h (CPU) |

### 9.6 关于 n=100 DRL 结果的重要说明

n=100 的最新 DRL 结果使用的是 `epoch-85.pt`（而非 epoch-99），且 f₁ 值远高于早期实验。这表明：

1. **可能 100 epoch 未充分收敛**（特别是 n=100 规模更大）
2. **200 epoch 实验已启动**：`train_raw_100_200ep` 正在运行
3. **早期实验的 n=100 结果**（weekly summary 中的 f₁=4339–5612）可能来自不同 checkpoint 或不同训练配置

---

## 10. 可视化系统

### 10.1 可视化脚本清单

| 脚本 | 功能 | 输出 |
|------|------|------|
| `visualization/pareto_inference.py` | DRL Pareto 推理 + 绘图 | `.pkl` + `.png` |
| `visualization/plot_convergence.py` | 多 λ 收敛曲线 | `convergence.png` |
| `visualization/plot_moead_comparison.py` | DRL vs MOEA/D 对比图 | `moead_vs_drl_*.png` |
| `visualization/plot_pareto_comparison.py` | 多实验 Pareto 叠加 | `pareto_comparison.png` |
| `visualization/plot_routes.py` | 不同 λ 的路线可视化 | `route_comparison_*.png` |
| `visualization/plot_agh50_200ep.py` | 200 epoch n=50 分析 | 收敛 + Pareto |

### 10.2 收敛图说明

`plot_convergence.py` 加载多个 epoch 的 checkpoint，在 5 个 λ 值下评估 cost：
- λ = (0.0, 1.0), (0.3, 0.7), (0.5, 0.5), (0.7, 0.3), (1.0, 0.0)
- 使用 `shared_test_50.pkl` 确保与 Pareto 推理一致
- 缓存：`paretofront/convergence_data.pkl`

### 10.3 对比图说明

`plot_moead_comparison.py`：

```bash
python visualization/plot_moead_comparison.py \
    --graph_size 50 \
    --drl_path paretofront/pareto_results_50_raw_101w.pkl \
    --moead_path paretofront/moead_results_50_exp1.pkl \
    --exp_name raw_101w
```

### 10.4 生成的图表文件

| 目录 | 内容 |
|------|------|
| `paretofront/*.png` | Pareto 对比、收敛曲线、路线图 |
| `images/convergence/` | 收敛曲线图 |
| `images/raw_101w/` | DRL 101 权重 Pareto 图 |
| `images/comparison_raw_101w/` | DRL vs MOEA/D 对比图 |
| `images/agh50_200ep/` | 200 epoch 分析图 |

---

## 11. SLURM 集群配置

### 11.1 通用配置

| 参数 | 值 |
|------|-----|
| 分区 | gpu_a100 |
| GPU | NVIDIA A100 × 1 |
| Conda 环境 | hhc |
| Python 选项 | `-u`（unbuffered） |

### 11.2 训练脚本

| 脚本 | n | epochs | run_name | 输出目录 |
|------|---|--------|----------|---------|
| `scripts/train_no_norm_50.sh` | 50 | 100 | train_raw_50 | outputs/agh_50/ |
| `scripts/train_no_norm_100.sh` | 100 | 100 | train_raw_100 | outputs/agh_100/ |
| `scripts/train_raw_50_200ep.sh` | 50 | 200 | train_raw_50_200ep | outputs/agh_50/ |
| `scripts/train_raw_100_200ep.sh` | 100 | 200 | train_raw_100_200ep | outputs/agh_100/ |

### 11.3 推理脚本

| 脚本 | 方法 | n | 说明 |
|------|------|---|------|
| `scripts/comparison_drl_50.sh` | DRL | 50 | 自动检测最新 checkpoint，101 权重 |
| `scripts/comparison_drl_100.sh` | DRL | 100 | 同上 |
| `scripts/comparison_moead_50.sh` | MOEA/D | 50 | 101 权重，500 代 |
| `scripts/comparison_moead_100.sh` | MOEA/D | 100 | 同上 |
| `scripts/comparison_pymoo_50.sh` | pymoo MOEA/D | 50 | 可选 |
| `scripts/comparison_pymoo_100.sh` | pymoo MOEA/D | 100 | 可选 |

### 11.4 可视化脚本

| 脚本 | 功能 |
|------|------|
| `scripts/run_comparison_plot.sh` | 一键生成 DRL vs MOEA/D 对比图 |
| `scripts/run_moead_comparison_plot.sh` | MOEA/D 对比绘图 |
| `scripts/plot_convergence.sh` | 收敛曲线 |
| `scripts/plot_convergence_200ep.sh` | 200 epoch 收敛 |
| `scripts/convergence_job.sh` | 收敛图 SLURM job |
| `scripts/pareto_inference_job.sh` | Pareto 推理 |

---

## 12. 完整文件清单

### 12.1 核心算法文件

| 文件 | 行数约 | 功能 |
|------|--------|------|
| `run.py` | 226 | 训练入口：模型初始化、优化器、训练循环 |
| `train.py` | 379 | 训练逻辑：train_epoch, validate, rollout, train_batch_agh |
| `options.py` | 108 | CLI 参数定义 |
| `reinforce_baselines.py` | 444 | 基线：NoBaseline, Exponential, Critic, Rollout, Warmup |
| `moead.py` | 1055 | MOEA/D 完整实现 |
| `hhc_problem_pymoo.py` | — | pymoo 封装（可选） |
| `generate_test_data.py` | 65 | 共享测试数据生成 |
| `generate_data.py` | — | 数据生成（遗留，未使用） |
| `eval.py` | — | 评估：beam search, sampling |
| `smoke_test.py` | — | 快速测试 |

### 12.2 网络模块（nets/）

| 文件 | 功能 |
|------|------|
| `nets/attention_model.py` | Attention Model + WE-Add λ 嵌入 |
| `nets/graph_encoder.py` | GraphAttentionEncoder（含 W_lambda 层） |
| `nets/pointer_network.py` | LSTM Pointer Network（备选模型） |
| `nets/critic_network.py` | Critic 网络（未使用） |

### 12.3 问题定义（problems/agh/）

| 文件 | 功能 |
|------|------|
| `problems/__init__.py` | 问题注册 |
| `problems/agh/__init__.py` | 空 |
| `problems/agh/problem_agh.py` | AGH 问题：cost 计算, 数据集, 状态初始化 |
| `problems/agh/state_agh.py` | AGH 状态：visited mask, time windows, fleet, masking |
| `problems/agh/data_read.py` | Fleet info 定义和 pickle 生成 |
| `problems/agh/data_find.py` | 数据验证工具 |
| `problems/agh/read.py` | 遗留 fleet_info 加载器 |
| `problems/agh/arrival_prob.npy` | 到达概率分布 |
| `problems/agh/coordinates.pkl` | 节点坐标 |
| `problems/agh/distance.pkl` | 101×101 距离矩阵 |
| `problems/agh/fleet_info.pkl` | 车队信息 |

### 12.4 工具函数（utils/）

| 文件 | 功能 |
|------|------|
| `utils/__init__.py` | 导出 functions |
| `utils/functions.py` | load_problem, torch_load_cpu, move_to, load_model |
| `utils/beam_search.py` | Beam search |
| `utils/boolmask.py` | Bool/long mask 转换 |
| `utils/data_utils.py` | save_dataset, load_dataset |
| `utils/lexsort.py` | 字典序排序 |
| `utils/log_utils.py` | Logger, log_values |
| `utils/monkey_patch.py` | Optimizer load_state_dict patch |
| `utils/tensor_functions.py` | compute_in_batches |

### 12.5 可视化（visualization/）

| 文件 | 功能 |
|------|------|
| `visualization/pareto_inference.py` | DRL Pareto 推理 + 绘图 |
| `visualization/plot_convergence.py` | 多 λ 收敛曲线 |
| `visualization/plot_moead_comparison.py` | DRL vs MOEA/D 对比图 |
| `visualization/plot_pareto_comparison.py` | 多实验 Pareto 叠加 |
| `visualization/plot_routes.py` | 路线可视化 |
| `visualization/plot_agh50_200ep.py` | 200 epoch 分析图 |

### 12.6 验证脚本

| 文件 | 功能 |
|------|------|
| `scripts/verify_drl_moead_consistency.py` | 验证 DRL vs MOEA/D 测试集/距离/fleet_info 一致性 |

### 12.7 结果数据（paretofront/）

| 文件 | 说明 |
|------|------|
| `shared_test_50.pkl` | 共享测试实例 n=50, 1000 个 |
| `shared_test_100.pkl` | 共享测试实例 n=100, 1000 个 |
| `pareto_results_50_exp1.pkl` | DRL n=50 早期实验 (11 权重) |
| `pareto_results_50_raw_101w.pkl` | DRL n=50 最新实验 (101 权重) |
| `pareto_results_50_raw_200ep.pkl` | DRL n=50 200 epoch |
| `pareto_results_50_norm_101w.pkl` | DRL n=50 归一化版本 |
| `pareto_results_100_exp1.pkl` | DRL n=100 早期实验 |
| `pareto_results_100_raw_101w.pkl` | DRL n=100 最新实验 (101 权重) |
| `pareto_results_100_raw_200ep.pkl` | DRL n=100 200 epoch |
| `moead_results_50_exp1.pkl` | MOEA/D n=50 (初步版本) |
| `moead_results_100_exp1.pkl` | MOEA/D n=100 (正式版本) |
| `convergence_data.pkl` | 收敛曲线缓存数据 |

---

## 13. Git 提交历史

### 完整提交记录（由新到旧）

| Hash | 描述 |
|------|------|
| `60c9a03` | 删除多余的 AGH 问题算法 |
| `86847c5` | 生成新的 101 gates 数据 |
| `eb050dd` | 修复归一化比例因子（f2 → f1 量级）|
| `3e22911` | 添加 EMA cost 归一化 |
| `837d1f4` | 重构 fleet 约束 + MOEA/D 自定义权重 |
| `ba38c1c` | 所有可视化图片 |
| `10d9ef6` | 添加 Fleet 3/5/6 combo 约束 |
| `1229fcf` | 添加 pareto_inference + 收敛/路线绘图 |
| `d6f76c1` | 添加 MOEA/D 算法 |
| `6f7fcfe` | 重构问题定义到子目录 |
| `73d42ed` | 引入通配符目录 + 更新基线 |
| `f76812f` | 重构 PCTSP/OP/VRP/TSP + 添加可视化 |
| `57af683` | 重做 TSP/PCTSP/OP/VRP 定义 |
| `e81ea01` | 重构问题定义 + 添加新求解器 |
| `5665091` | 问题定义移入新子目录 |
| `f112ca7` | GPU 优化 + 修复 |
| `67f061e` | GPU 优化 + 多目标权重支持 |
| `f3a1698` | 移除 POMO baseline |
| `72de17b` | 添加 W_lambda 层到 GraphAttentionEncoder |
| `84619a4` | 添加 lambda_dim + W_lambda 层 |
| `398ca5b` | 添加 tsp_gpu 环境配置 |
| `a8efcc4` | 添加 README 实验记录 |
| `2de7be8` | 修改问候语 |
| `1d3e08c` | 删除 .idea 目录 |
| `5fb11bf` | 初始化仓库 |

---

## 14. 训练运行记录

### 14.1 所有训练运行

| 运行目录 | n | epochs | 类型 | 状态 |
|----------|---|--------|------|------|
| `outputs/agh_50/run_20260213T165016/` | 50 | — | 早期实验 | 已完成 |
| `outputs/agh_50/run_20260213T214438/` | 50 | 100 | 早期实验 (epoch-54~100) | 已完成 |
| `outputs/agh_50/run_20260225T114324/` | 50 | — | 实验 | 已完成 |
| `outputs/agh_50/run_20260225T160526/` | 50 | — | 实验 | 已完成 |
| `outputs/agh_50/run_20260226T190716/` | 50 | — | 实验 | 已完成 |
| `outputs/agh_50/train_norm_50_20260227T135722/` | 50 | — | 归一化版本 | 已完成 |
| `outputs/agh_50/train_no_norm_50_20260228T005602/` | 50 | 100 | raw cost | 已完成 |
| `outputs/agh_50/train_raw_50_20260228T011907/` | 50 | 100 | raw cost | 已完成 |
| `outputs/agh_50/train_raw_50_200ep_20260228T162007/` | 50 | 200 | raw cost 200ep | 已完成 |
| `outputs/agh_100/run_20260214T095740/` | 100 | 100 | 早期实验 (epoch-0~99) | 已完成 |
| `outputs/agh_100/run_20260225T114324/` | 100 | — | 实验 | 已完成 |
| `outputs/agh_100/run_20260226T190716/` | 100 | — | 实验 | 已完成 |
| `outputs/agh_100/train_norm_100_20260227T140051/` | 100 | — | 归一化版本 | 已完成 |
| `outputs/agh_100/train_no_norm_100_20260228T005602/` | 100 | 100 | raw cost | 已完成 |
| `outputs/agh_100/train_raw_100_20260228T011907/` | 100 | 100 | raw cost | 已完成 |
| `outputs/agh_100/train_raw_100_200ep_20260228T162008/` | 100 | 200 | raw cost 200ep | 可能进行中 |

### 14.2 训练配置变迁

项目经历了多次 cost 函数变化：

1. **最初**：`cost = f₁`（仅距离）
2. **添加多目标权重**：`cost = w_dist · f₁ + w_wait · f₂`
3. **WE-Add**：`cost = λ₁ · f₁ + λ₂ · f₂`（随机采样 λ）
4. **EMA 归一化尝试**：`cost = λ₁ · f₁/s₁ + λ₂ · f₂/s₂`（已弃用）
5. **固定比例归一化尝试**：`cost = λ₁ · f₁ + λ₂ · (f₂ × 30)`（已弃用）
6. **当前**：`cost = λ₁ · f₁ + λ₂ · f₂`（原始加权和，无归一化）

**弃用归一化的原因**：ObjectiveNormalizer 引入了额外的不稳定性，且原始加权和在实验中表现已足够好。

---

## 15. 已知问题与改进方向

### 15.1 当前已知问题

| 问题 | 严重程度 | 详情 |
|------|---------|------|
| n=100 DRL 100ep 可能未充分收敛 | 高 | f₁ 远高于预期（~6200 vs 早期 ~4300） |
| n=50 MOEA/D 缺少正式 1000 实例结果 | 中 | 当前仅有 5 实例的初步测试 |
| Encoder 不接收距离矩阵 | 低 | 仅通过 loc_embedding 间接利用距离 |
| MOEA/D λ 调控力弱 | 低 | f₁ 跨度窄，因为 λ 仅影响 Tchebycheff 替换 |
| DRL f₂ 在 n=100 时远大于 MOEA/D | 中 | DRL 在高 λ₁ 时 f₂ 达 1288 vs MOEA/D 447 |
| 大量未使用的遗留文件 | 低 | generate_data.py、MoVRPTW.py 等 |

### 15.2 可能的改进方向

1. **增加训练 epoch**：至少 200 epoch，验证收敛（已启动 200ep 实验）
2. **学习率调度**：当前 lr_decay=1.0 无衰减，可尝试余弦衰减或 warmup
3. **距离信息增强**：考虑将距离矩阵或距离特征输入 Encoder
4. **MOEA/D 增强**：增大种群、增加代数、引入 λ 条件化搜索算子
5. **归一化重新考虑**：虽然当前弃用，但 f₁ 和 f₂ 量级差异大（~3000 vs ~100），可能影响 λ 的线性调控效果
6. **Hypervolume 指标**：已在 MOEA/D 中实现，需在 DRL 侧也计算
7. **代码清理**：移除遗留文件，统一命名规范

### 15.3 重要设计决策记录

| 决策 | 理由 |
|------|------|
| 使用 WE-Add 而非 WE-Concat | WE-Add 更简单，效果在论文中验证过 |
| 原始加权和无归一化 | 避免 EMA 归一化带来的不稳定性 |
| Rollout baseline（非 Critic） | AGH 多 fleet 问题适合 rollout |
| 6 fleet 按固定顺序解码 | 与原始 AGH 论文一致 |
| Combo 约束仅对 combo-need 患者 | 与 DRL masking 对齐后的正确行为 |
| λ₁ ~ Uniform(0,1) | 确保 Pareto 前沿上所有偏好都被覆盖 |

---

## 16. 复现指南

### 16.1 环境准备

```bash
conda activate hhc
cd /home/ltan/HHC
```

### 16.2 完整实验流水线

```bash
# 1. 生成共享测试数据
python generate_test_data.py --n_instances 1000 --seed 1234

# 2. 训练 DRL 模型
sbatch scripts/train_no_norm_50.sh    # n=50, 100 epochs
sbatch scripts/train_no_norm_100.sh   # n=100, 100 epochs
# 或 200 epochs：
sbatch scripts/train_raw_50_200ep.sh
sbatch scripts/train_raw_100_200ep.sh

# 3. DRL Pareto 推理（训练完成后）
sbatch scripts/comparison_drl_50.sh
sbatch scripts/comparison_drl_100.sh

# 4. MOEA/D 运行
sbatch scripts/comparison_moead_50.sh
sbatch scripts/comparison_moead_100.sh

# 5. 生成对比图
bash scripts/run_comparison_plot.sh

# 6. 收敛曲线（可选）
sbatch scripts/plot_convergence.sh
```

### 16.3 单次训练命令

```bash
python -u run.py \
    --n_epochs 100 \
    --graph_size 50 \
    --problem agh \
    --baseline rollout \
    --run_name train_raw_50
```

### 16.4 单次 MOEA/D 命令

```bash
python -u moead.py \
    --graph_size 50 \
    --filename paretofront/shared_test_50.pkl \
    --n_instances 1000 \
    --n_weights 101 \
    --n_gen 500 \
    --seed 1234
```

### 16.5 单次 Pareto 推理命令

```bash
python -u visualization/pareto_inference.py \
    --load_path outputs/agh_50/<run_dir>/epoch-99.pt \
    --graph_size 50 \
    --val_size 1000 \
    --val_dataset paretofront/shared_test_50.pkl \
    --seed 1234 \
    --exp_name raw_101w
```

---

## 附录 A：完整训练参数（args.json 示例）

```json
{
  "problem": "agh",
  "graph_size": 50,
  "batch_size": 64,
  "epoch_size": 12800,
  "val_size": 1000,
  "val_dataset": null,
  "model": "attention",
  "embedding_dim": 128,
  "hidden_dim": 128,
  "n_encode_layers": 3,
  "tanh_clipping": 10.0,
  "normalization": "batch",
  "optimizer": "Adam",
  "lr_model": 0.0001,
  "lr_critic": 0.0001,
  "lr_decay": 1.0,
  "n_epochs": 200,
  "seed": 123456,
  "max_grad_norm": 1.0,
  "baseline": "rollout",
  "bl_alpha": 0.05,
  "bl_warmup_epochs": 0,
  "eval_batch_size": 1000,
  "checkpoint_encoder": false,
  "shrink_size": null,
  "log_step": 50,
  "checkpoint_epochs": 1,
  "fine_tune": false,
  "wo_time": false,
  "rnn_time": false,
  "use_cuda": true
}
```

---

## 附录 B：Conda 环境关键依赖

| 包 | 用途 |
|----|------|
| python | 3.x |
| pytorch + cuda | 模型训练 |
| numpy | 数据处理 |
| scipy | t 检验（rollout baseline） |
| matplotlib | 可视化 |
| tqdm | 进度条 |
| tensorboard_logger | TensorBoard（已禁用） |
| pymoo | 可选 MOEA/D 封装 |

---

**本文档于 2026-03-02 由 AI 助手自动生成，基于对项目全部源代码、训练日志、实验结果和历史总结的详细审查。**
