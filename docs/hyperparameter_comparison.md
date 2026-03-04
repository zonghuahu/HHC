# DRL vs MOEA/D 超参数对照表

## 主表

| 参数名称 | DRL 值 | MOEA/D 值 |
|----------|--------|-----------|
| **网络架构** | | |
| Encoder 类型 | GraphAttentionEncoder | N/A |
| Decoder 类型 | Attention-based (自回归) | N/A |
| embedding_dim | 128 | N/A |
| hidden_dim | 128 | N/A |
| n_encode_layers | 3 | N/A |
| n_heads (attention heads) | 8 | N/A |
| feed_forward_hidden | 512 | N/A |
| normalization (encoder) | batch | N/A |
| tanh_clipping | 10.0 | N/A |
| mask_inner | True | N/A |
| mask_logits | True | N/A |
| lambda_dim | 2 | N/A |
| **训练参数** | | |
| learning rate (lr_model) | 1e-4 | N/A |
| lr_critic | 1e-4 | N/A |
| lr_decay | 1.0 | N/A |
| optimizer | Adam | N/A |
| batch_size | 64 | N/A |
| epoch_size (每 epoch 实例数) | 12800 | N/A |
| n_epochs | 200 | N/A |
| max_grad_norm | 1.0 | N/A |
| seed | 123456 | 1234 |
| **REINFORCE / Baseline** | | |
| baseline 类型 | rollout | N/A |
| greedy rollout for baseline | Yes | N/A |
| bl_alpha (t-test 显著性) | 0.05 | N/A |
| bl_warmup_epochs | 0 | N/A |
| exp_beta | 0.8 | N/A |
| **多目标相关** | | |
| λ 采样方式 | 每 batch 均匀采样 U[0,1] | 均匀网格 0, 0.01, ..., 1.0 |
| 权重嵌入方式 | WE-Add (W_lambda 线性层加和) | N/A |
| WE-Concat | No | N/A |
| λ 数量 (推断/评估) | 101 | 101 |
| cost 公式 | λ₁·f₁ + λ₂·f₂ | Tchebycheff 分解 |
| **解码方式** | | |
| 训练 decode_type | sampling | N/A |
| 验证/推断 decode_type | greedy | N/A |
| temperature (softmax) | 1.0 | N/A |
| **归一化** | | |
| 层归一化 (batch/instance) | batch | N/A |
| min-max cost 归一化 | No | N/A |
| EMA cost 归一化 | No | N/A |
| **数据** | | |
| 训练数据 | 随机生成 (epoch_size) | N/A |
| 验证/测试实例数 | 1000 | 1000 |
| 测试数据集 | shared_test_*.pkl | shared_test_*.pkl |
| **MOEA/D 专用** | | |
| 权重向量数量 (n_weights) | N/A | 101 |
| 邻域大小 (T) | N/A | 3 |
| 代数 (n_gen) | N/A | 500 |
| 种群大小 | N/A | = n_weights (101) |
| 交叉算子 | N/A | Order Crossover (OX) |
| 交叉概率 | N/A | 1.0 (始终应用) |
| 子代选择 | N/A | 0.5 (child1/child2 随机) |
| 变异算子 | N/A | swap, relocate, reverse_segment (各 1/3 概率) |
| 变异概率 (mutation_rate) | N/A | 0.3 |
| 分解方式 | N/A | Tchebycheff |
| 每实例独立优化 | N/A | Yes |
| 初始解构造 | N/A | NN sorted + NN/random 交替 |

---

## 源代码位置索引

### DRL

| 参数 | 文件 | 行号 |
|------|------|------|
| embedding_dim, hidden_dim, n_encode_layers | options.py | 22-24 |
| tanh_clipping | options.py | 25-26 |
| normalization | options.py | 27 |
| optimizer | options.py | 28 |
| lr_model, lr_critic, lr_decay | options.py | 30-32 |
| n_epochs | options.py | 34 |
| seed | options.py | 35 |
| max_grad_norm | options.py | 36-37 |
| exp_beta | options.py | 39-40 |
| baseline | options.py | 42-43 |
| bl_alpha | options.py | 44-45 |
| bl_warmup_epochs | options.py | 46-48 |
| batch_size | options.py | 14 |
| epoch_size | options.py | 15 |
| val_size | options.py | 16-17 |
| eval_batch_size | options.py | 48-49 |
| n_heads | nets/attention_model.py | 66 |
| temp (temperature) | nets/attention_model.py | 78 |
| mask_inner, mask_logits | run.py | 74-75 |
| lambda_dim | run.py | 82 |
| W_lambda (WE-Add) | nets/graph_encoder.py | 260-277 |
| feed_forward_hidden | nets/graph_encoder.py | 253, 264 |
| λ 采样 (torch.rand) | train.py | 231-232 |
| cost 公式 | train.py | 108, 293, 325 |
| training decode sampling | train.py | 177, 228 |
| validation greedy | train.py | 36 |
| baseline eval_agh greedy | reinforce_baselines.py | 274 |
| num_lambdas (推断) | visualization/pareto_inference.py | 37, 185 |
| val_size | visualization/pareto_inference.py | 180 |
| pareto inference batch_size | visualization/pareto_inference.py | 51 |

### MOEA/D

| 参数 | 文件 | 行号 |
|------|------|------|
| n_weights | moead.py | 960 |
| n_weights (scripts) | scripts/run_moead.sh | 16, 28 |
| n_gen | moead.py | 965 |
| T | moead.py | 967 |
| mutation_rate | moead.py | 969 |
| n_instances | moead.py | 956 |
| seed | moead.py | 971 |
| n_gen, T 传入 | moead.py | 1010-1013 |
| generate_weight_vectors | moead.py | 709-716 |
| Tchebycheff | moead.py | 735-738, 860-868 |
| crossover_ox | moead.py | 617-639, 760-777 |
| crossover 应用 | moead.py | 842-844 |
| child 选择 0.5 | moead.py | 846 |
| apply_mutation | moead.py | 741-758 |
| mutation_rate/3 分配 | moead.py | 750-756 |
| population 初始化 | moead.py | 808-825 |
| per-instance 循环 | moead.py | 803-874 |
| filename | scripts/run_moead.sh | 14, 23 |
