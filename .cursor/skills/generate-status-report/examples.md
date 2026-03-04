# Example Status Report

Example output structure (values are illustrative):

```markdown
# HHC Project Status Report
Generated: 2026-02-28T12:00:00

## 1. Core File Modification Record (vs git)

 M moead.py
 M nets/attention_model.py
 M train.py
?? objective_normalizer.py

## 2. moead.py `is_visit_feasible` (Full Code)

```python
def is_visit_feasible(svc_start, dur_p, tw_left_p, tw_right_p,
                      fleet_id, patient_need):
    # C1: Basic time window
    if svc_start + dur_p > tw_right_p + 1e-5:
        return False
    # C2: Combo-patient late-arrival constraint
    if fleet_id in COMBO_NEED:
        ...
    return True
```

## 3. attention_model.py `forward()` Cost Calculation

cost = λ₁·f₁ + λ₂·f₂ (raw, no normalization)

## 4. outputs/ Directory

| Run Path | Cost Version |
|----------|--------------|
| agh_50/train_norm_50_20260227T135722 | norm |
| agh_50/train_raw_50_20260228T011907 | raw |
| agh_100/run_20260214T095740 | unknown |

## 5. paretofront/ Result Files

| File | Source |
|------|--------|
| shared_test_50.pkl | Shared test instances (n=50) |
| shared_test_100.pkl | Shared test instances (n=100) |
| pareto_results_50_raw_101w.pkl | DRL Pareto (raw cost, 101 gates) |
| moead_results_50_exp1.pkl | MOEA/D evaluation |

## 6. Running SLURM Tasks

 JOBID  USER  ST  NAME
 12345  ltan  R  train_norm_50

## 7. Known Issues & TODO

- DRL vs MOEA/D must use same shared_test_*.pkl
- state_agh masking must match moead is_visit_feasible
```
