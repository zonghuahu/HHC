# DRL Cost / Scalarization 分析文档

## 1. Cost 计算的完整调用链

### 1.1 训练主循环入口

| 步骤 | 文件 | 函数 | 行号 | 输入 | 输出 |
|------|------|------|------|------|------|
| 1 | train.py | train_epoch | 150-223 | model, optimizer, baseline, ... | — |
| 2 | train.py | train_batch_agh | 184 | batch | — |
| 3 | train.py | train_batch_agh | 231-232 | — | lambda_vector [bs, 2] |

### 1.2 λ 采样

| 步骤 | 文件 | 函数 | 行号 | 说明 |
|------|------|------|------|------|
| λ 采样 | train.py | train_batch_agh | 231-232 | `lam = torch.rand(bs, 1, device=opts.device)`；`lambda_vector = torch.cat([lam, 1 - lam], dim=1)` |

- **形状**：`lambda_vector` 为 `[batch_size, 2]`，每行 `[λ₁, λ₂]`，λ₁ ~ U[0,1]，λ₂ = 1 - λ₁
- **传入**：作为 `model(..., lambda_vector=lambda_vector)` 和 `baseline.eval_agh(..., lambda_vector, opts)` 的参数

### 1.3 f₁、f₂ 计算位置

| 变量 | 文件 | 函数 | 行号 | 说明 |
|------|------|------|------|------|
| f1 | problems/agh/problem_agh.py | AGH.get_costs | 50 | `f1 = batch_distance.gather(1, distance_index).sum(1)`，总行驶距离 |
| f2 | problems/agh/problem_agh.py | AGH.get_costs | 51 | `f2 = total_wait`，总患者等待时间 |
| total_wait | problems/agh/problem_agh.py | AGH.get_costs | 43-44 | `wait_i = torch.clamp(arrival_time - tw_left_i, min=0)`，逐节点累加 |

**调用链**：`model.forward()` → `self.problem.get_costs(input, pi)` → 返回 `(f1, f2), None`

### 1.4 标量化 cost 组合（完整代码行）

| 位置 | 文件 | 行号 | 代码 |
|------|------|------|------|
| 训练 policy | train.py | 293 | `fleet_cost = lambda_vector[:, 0] * f1 + lambda_vector[:, 1] * f2` |
| 训练 loss | train.py | 315-318 | `loss = ((fleet_cost_list[0] - bl_cost_list[0]) * log_likelihood_list[0]).mean()` + ... |
| 日志 | train.py | 325 | `fleet_cost_together = lambda_vector[:, 0] * total_f1 + lambda_vector[:, 1] * total_f2` |

### 1.5 cost 的后续使用

| 用途 | 文件 | 行号 | 说明 |
|------|------|------|------|
| REINFORCE loss | train.py | 315-318 | `loss = ((fleet_cost - bl_cost) * log_likelihood).mean()`，advantage = fleet_cost - bl_cost |
| log_values | train.py | 327 | `log_values(fleet_cost_together, ...)` 用于打印 avg_cost |
| log_utils | utils/log_utils.py | 35 | `avg_cost = cost.mean().item()` |

---

## 2. Validation / Inference 中的 cost 计算

### 2.1 是否复用训练中的 cost 函数

**是**。Validation 和 pareto_inference 都通过 `model(...)` 调用 `problem.get_costs`，与训练阶段共用同一套 f₁、f₂ 计算逻辑。

### 2.2 Validation 阶段

| 步骤 | 文件 | 函数 | 行号 | 说明 |
|------|------|------|------|------|
| 入口 | train.py | validate | 24-31 | 调用 `rollout(model, dataset, opts)` |
| λ | train.py | rollout | 48-53 | `lambda_vector is None` 时使用 `lv = [[0.5, 0.5]]` |
| model 输出 | train.py | rollout | 105-106 | `f1, f2, _, serve_time = model(...)`；`scalarized_cost = lv[:, 0] * f1 + lv[:, 1] * f2` |
| 汇总 | train.py | validate | 28-29 | `cost = cost.sum(1)`（6 个 fleet 的 cost 求和） |

### 2.3 Pareto Inference 阶段

| 步骤 | 文件 | 函数 | 行号 | 说明 |
|------|------|------|------|------|
| λ 扫描 | visualization/pareto_inference.py | pareto_inference | 42-47 | `lambdas = np.linspace(0, 1, num_lambdas)`；`lv = [[l1, l2]]` |
| model 输出 | visualization/pareto_inference.py | pareto_inference | 113-116 | `f1, f2, _, serve_time = model(...)`；**不计算 scalarized cost** |
| 存储 | visualization/pareto_inference.py | pareto_inference | 131-137 | 仅保存 `f1_mean`, `f1_std`, `f2_mean`, `f2_std` |

### 2.4 Pareto 绘图阶段

| 步骤 | 文件 | 函数 | 行号 | 说明 |
|------|------|------|------|------|
| 绘图 | visualization/pareto_inference.py | plot_pareto | 148-167 | 使用 `f1_means`, `f2_means`，**直接用 (f₁, f₂)** |
| 对比图 | visualization/plot_moead_comparison.py | plot_comparison | 34-45 | 同样使用 `f1_mean`, `f2_mean`，**不涉及 scalarized cost** |

**结论**：Pareto 推断和绘图阶段只使用原始 (f₁, f₂)，不进行标量化。

---

## 3. Baseline（REINFORCE baseline）中的 cost 计算

### 3.1 Greedy rollout 是否计算 scalarized cost

**是**。`eval_agh` 中在得到 f1、f2 后显式计算 `bl_fleet_cost = λ₁·f1 + λ₂·f2`。

### 3.2 Baseline 使用的 λ

**与 policy 相同**。`train_batch_agh` 将当前 batch 的 `lambda_vector` 传入 `baseline.eval_agh(..., lambda_vector, opts)`，baseline 使用同一 λ 做 greedy rollout。

### 3.3 Baseline cost 计算代码位置

| 文件 | 函数 | 行号 |
|------|------|------|
| reinforce_baselines.py | RolloutBaseline.eval_agh | 266-346 |

---

## 4. 当前 cost 相关代码片段汇总

### train.py

```python
# train.py:24-31  validate
def validate(model, dataset, opts):
    """验证模型性能，使用固定 lambda=[0.5, 0.5] 评估标量化成本。"""
    print('Validating...')
    cost = rollout(model, dataset, opts)
    if model.is_agh:
        cost = cost.sum(1)
    avg_cost = cost.mean()
    ...
```

```python
# train.py:48-53  rollout λ
            if lambda_vector is None:
                lv = torch.tensor([[0.5, 0.5]], device=opts.device).expand(bs, -1)
            else:
                lv = lambda_vector.to(opts.device)
                if lv.dim() == 1:
                    lv = lv.unsqueeze(0).expand(bs, -1)
```

```python
# train.py:105-108  rollout scalarized cost
                with torch.no_grad():
                    f1, f2, _, serve_time = model(move_to(fleet_bat, opts.device), lambda_vector=lv)
                scalarized_cost = lv[:, 0] * f1 + lv[:, 1] * f2
                bat_cost.append(scalarized_cost.data.cpu().view(-1, 1))
```

```python
# train.py:231-232  train_batch_agh λ 采样
    lam = torch.rand(bs, 1, device=opts.device)
    lambda_vector = torch.cat([lam, 1 - lam], dim=1)  # [batch_size, 2]
```

```python
# train.py:291-295  train_batch_agh policy cost
        f1, f2, log_likelihood, serve_time = model(move_to(fleet_bat, opts.device), lambda_vector=lambda_vector)

        fleet_cost = lambda_vector[:, 0] * f1 + lambda_vector[:, 1] * f2

        fleet_cost_list.append(fleet_cost)
```

```python
# train.py:310-318  train_batch_agh baseline & loss
    bl_cost_list = baseline.eval_agh(x, model.fleet_info, model.distance, lambda_vector, opts)
    ...
    loss = ((fleet_cost_list[0] - bl_cost_list[0]) * log_likelihood_list[0]).mean()
    for i in range(1, len(fleet_cost_list)):
        loss += ((fleet_cost_list[i] - bl_cost_list[i]) * log_likelihood_list[i]).mean()
    loss = loss / len(fleet_cost_list)
```

```python
# train.py:325-327  train_batch_agh log
    fleet_cost_together = lambda_vector[:, 0] * total_f1 + lambda_vector[:, 1] * total_f2
    if step % int(opts.log_step) == 0:
        log_values(fleet_cost_together, grad_norms, epoch, batch_id, step, log_likelihood_together, loss, 0, tb_logger, opts)
```

### problems/agh/problem_agh.py

```python
# problems/agh/problem_agh.py:16-53  get_costs
    @staticmethod
    def get_costs(dataset, pi):
        """
        计算路径的双目标成本：f1（总行驶距离）和 f2（总等待时间）。
        返回：((f1, f2), mask=None)
        """
        ...
        for i in range(pi.size(1)):
            arrival_time = cur_time + time_distance[:, i:i+1]
            tw_left_i = dataset['tw_left'][ids, pi[:, i:i+1]]
            wait_i = torch.clamp(arrival_time - tw_left_i, min=0) * (pi[:, i:i+1] != 0).float()
            total_wait += wait_i.squeeze(1)
            ...
        f1 = batch_distance.gather(1, distance_index).sum(1)
        f2 = total_wait

        return (f1, f2), None
```

### nets/attention_model.py

```python
# nets/attention_model.py:173-196  forward
    def forward(self, input, lambda_vector=None, return_pi=False):
        ...
        cost, mask = self.problem.get_costs(input, pi)

        ll = self._calc_log_likelihood(_log_p, pi, mask)

        if self.is_agh:
            f1, f2 = cost  # get_costs now returns (f1, f2)
            if return_pi:
                return f1, f2, ll, serve_time, pi
            else:
                return f1, f2, ll, serve_time
```

### reinforce_baselines.py

```python
# reinforce_baselines.py:330-336  eval_agh baseline cost
            with torch.no_grad():
                f1, f2, _, serve_time = self.model(
                    move_to(fleet_bat, opts.device),
                    lambda_vector=lambda_vector
                )
            bl_fleet_cost = lambda_vector[:, 0] * f1 + lambda_vector[:, 1] * f2
            bl_cost_list.append(bl_fleet_cost.detach())
```

```python
# reinforce_baselines.py:349-353  epoch_callback (validation cost)
        candidate_vals = rollout(model, self.dataset, self.opts).cpu().numpy()
        if model.is_agh:
            candidate_vals = candidate_vals.sum(1)
        candidate_mean = candidate_vals.mean()
```

### visualization/pareto_inference.py

```python
# visualization/pareto_inference.py:42-47  λ 扫描
    lambdas = np.linspace(0, 1, num_lambdas)
    ...
    for idx, l1 in enumerate(lambdas):
        l2 = 1.0 - l1
        lv = torch.tensor([[l1, l2]], dtype=torch.float32, device=device)
```

```python
# visualization/pareto_inference.py:112-116  model 输出，无 scalarization
                with torch.no_grad():
                    f1, f2, _, serve_time = model(move_to(fleet_bat, device), lambda_vector=lv_batch)

                batch_f1 += f1
                batch_f2 += f2
```

```python
# visualization/pareto_inference.py:129-137  results 存储 (f1, f2 原始值)
        r = {
            'lambda': (l1, l2),
            'f1_mean': all_f1.mean().item(),
            'f1_std': all_f1.std().item(),
            'f2_mean': all_f2.mean().item(),
            'f2_std': all_f2.std().item(),
        }
        results.append(r)
```

### visualization/plot_moead_comparison.py

```python
# visualization/plot_moead_comparison.py:34-45  直接用 f1_mean, f2_mean
    drl_f1 = np.array([r['f1_mean'] for r in drl_results])
    drl_f2 = np.array([r['f2_mean'] for r in drl_results])
    ...
    ax.plot(drl_f1[sort_idx], drl_f2[sort_idx], '-', ...)
```

### utils/log_utils.py

```python
# utils/log_utils.py:34-35  log_values
def log_values(cost, grad_norms, epoch, batch_id, step, log_likelihood, reinforce_loss, bl_loss, tb_logger, opts):
    avg_cost = cost.mean().item()
```
