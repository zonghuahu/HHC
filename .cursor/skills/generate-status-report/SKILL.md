---
name: generate-status-report
description: Generates a complete project status report for syncing with other AI assistants. Use when the user asks to sync project status, hand off context to another AI, or when starting a new conversation window and needs a full project summary.
---

# Generate Status Report

Generate a Markdown report for AI-to-AI handoff. Output is directly pasteable.

## Report Template

```markdown
# HHC Project Status Report
Generated: [ISO timestamp]

## 1. Core File Modification Record (vs git)

[Git status output]

## 2. moead.py `is_visit_feasible` (Full Code)

```python
[full function body]
```

## 3. attention_model.py `forward()` Cost Calculation

[Code block showing cost calculation logic]

## 4. outputs/ Directory

| Run Path | Cost Function Version |
|----------|----------------------|
| [path] | [norm|no_norm|raw|unknown] |

## 5. paretofront/ Result Files

| File | Source |
|------|--------|
| [file] | [description] |

## 6. Running SLURM Tasks

[Output of squeue or "None" if empty]

## 7. Known Issues & TODO

[From user memory, TODO comments, or "None" if not documented]
```

## Generation Steps

### Step 1: Git status
Run: `git -C /home/ltan/HHC status --short`

### Step 2: `is_visit_feasible`
Read `moead.py`, extract the full `is_visit_feasible` function (lines 91–113). Include all docstring and logic.

### Step 3: Cost calculation
Read `nets/attention_model.py` `forward()` method. Extract the AGH cost block:
```python
if self.is_agh:
    f1, f2, mask = self.problem.get_costs(input, pi, return_components=True)
    if lambda_vector is not None:
        cost = lambda_vector[:, 0] * f1 + lambda_vector[:, 1] * f2
    else:
        cost = f1
```
Note any normalization or variant in the current code.

### Step 4: outputs/ runs
- List all `outputs/agh_50/` and `outputs/agh_100/` subdirs
- Infer cost version from run name: `train_norm_*` → norm, `train_no_norm_*` → no_norm, `train_raw_*` → raw, `run_*` → unknown (check args.json if present)
- For each run, read `args.json` if exists; note `objective_normalizer` or `normalization` in args

### Step 5: paretofront/ pkl files
List all `.pkl` files. For each, infer source:
- `shared_test_*.pkl` → shared test instances (for DRL/MOEA/D comparison)
- `pareto_results_*` → DRL Pareto inference output (from `pareto_inference.py`)
- `moead_results_*` → MOEA/D evaluation output
- `convergence_data.pkl` → convergence plot data

### Step 6: SLURM tasks
Run: `squeue -u $USER` (or `squeue` if no user filter)

### Step 7: Issues & TODO
- Search for `TODO`, `FIXME`, `XXX` in core files
- Check user rules or memory for known issues
- If none found, write "None documented"

## Output

Emit the final report as Markdown. Do not embed it in code blocks unless the user asks to save to file.

## Reference

For an example report structure, see [examples.md](examples.md).
