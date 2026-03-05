# -*- coding: utf-8 -*-
"""
Compute and compare hypervolume (HV) for DRL and MOEA/D Pareto fronts.

- Loads DRL pareto_results_*.pkl and MOEA/D moead_results_*.pkl
- Extracts (f1, f2) points, filters to non-dominated solutions
- Uses common reference point: (max_f1 * 1.1, max_f2 * 1.1) over all points
- Computes HV for n=50 and n=100

Usage:
  python -u visualization/compute_hypervolume.py
  python -u visualization/compute_hypervolume.py --drl_exp tche --pareto_dir paretofront
  python -u visualization/compute_hypervolume.py --drl_exp_50 tmp50 --drl_exp_100 weadd101
"""
import os
import pickle
import argparse
import numpy as np

# Default paths (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARETO_DIR = os.path.join(PROJECT_ROOT, 'paretofront')


def load_results(filepath):
    """Load pickle results (list of dicts with f1_mean, f2_mean)."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def extract_points(results):
    """Extract (f1, f2) as Nx2 array from pareto_results format."""
    return np.array([[r['f1_mean'], r['f2_mean']] for r in results])


def filter_non_dominated(points):
    """
    Filter to non-dominated solutions (minimization).
    A point is non-dominated iff no other point is strictly better in both f1 and f2.
    """
    if len(points) == 0:
        return points
    # For each point i: dominated if exists j s.t. f1[j]<=f1[i] and f2[j]<=f2[i] and strict in at least one
    n = len(points)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is no worse in both and strictly better in at least one
            if (points[j, 0] <= points[i, 0] and points[j, 1] <= points[i, 1] and
                    (points[j, 0] < points[i, 0] or points[j, 1] < points[i, 1])):
                dominated[i] = True
                break
    return points[~dominated]


def hv_2d(points, ref_point):
    """
    Compute 2D hypervolume (area) with reference point.
    points: Nx2 array (f1, f2), minimization.
    ref_point: [ref_f1, ref_f2] - must be >= max of all points in each dimension.
    """
    ref = np.array(ref_point)
    # Filter points outside ref
    mask = (points[:, 0] < ref[0]) & (points[:, 1] < ref[1])
    pts = points[mask]
    if len(pts) == 0:
        return 0.0
    # Sort by f1 ascending
    idx = np.argsort(pts[:, 0])
    pts = pts[idx]
    # Sweepline: traverse right-to-left
    hv = 0.0
    prev_f1 = ref[0]
    for i in range(len(pts) - 1, -1, -1):
        f1, f2 = pts[i]
        width = prev_f1 - f1
        height = ref[1] - f2
        hv += width * height
        prev_f1 = f1
    return hv


def compute_hv_pymoo(points, ref_point):
    """Use pymoo HV if available."""
    try:
        from pymoo.indicators.hv import HV
        ind = HV(ref_point=np.array(ref_point))
        # pymoo HV expects minimization: higher is worse, ref should be nadir
        return ind(points)
    except ImportError:
        return None


def main():
    parser = argparse.ArgumentParser(description='Compute HV for DRL vs MOEA/D')
    parser.add_argument('--pareto_dir', type=str, default=PARETO_DIR)
    parser.add_argument('--drl_exp', type=str, default='tche',
                        help='DRL exp_name for both sizes: tche, exp1, weadd101, etc.')
    parser.add_argument('--drl_exp_50', type=str, default=None,
                        help='Override DRL exp for n=50')
    parser.add_argument('--drl_exp_100', type=str, default=None,
                        help='Override DRL exp for n=100')
    opts = parser.parse_args()

    sizes = [50, 100]
    rows = []

    for size in sizes:
        exp = opts.drl_exp_50 if size == 50 and opts.drl_exp_50 else (
              opts.drl_exp_100 if size == 100 and opts.drl_exp_100 else opts.drl_exp)
        drl_path = os.path.join(opts.pareto_dir, f'pareto_results_{size}_{exp}.pkl')
        moead_path = os.path.join(opts.pareto_dir, f'moead_results_{size}_exp1.pkl')

        if not os.path.isfile(drl_path):
            print(f"[WARN] DRL not found: {drl_path}")
            continue
        if not os.path.isfile(moead_path):
            print(f"[WARN] MOEA/D not found: {moead_path}")
            continue

        drl_raw = load_results(drl_path)
        moead_raw = load_results(moead_path)

        drl_pts = extract_points(drl_raw)
        moead_pts = extract_points(moead_raw)

        drl_nd = filter_non_dominated(drl_pts)
        moead_nd = filter_non_dominated(moead_pts)

        # Common reference: max over ALL points (both methods) * 1.1
        all_pts = np.vstack([drl_pts, moead_pts])
        ref_f1 = np.max(all_pts[:, 0]) * 1.1
        ref_f2 = np.max(all_pts[:, 1]) * 1.1
        ref_point = [ref_f1, ref_f2]

        # Compute HV (prefer pymoo, fallback to hand-written)
        hv_pymoo_drl = compute_hv_pymoo(drl_nd, ref_point)
        hv_pymoo_moead = compute_hv_pymoo(moead_nd, ref_point)

        if hv_pymoo_drl is not None and hv_pymoo_moead is not None:
            hv_drl = hv_pymoo_drl
            hv_moead = hv_pymoo_moead
            backend = 'pymoo'
        else:
            hv_drl = hv_2d(drl_nd, ref_point)
            hv_moead = hv_2d(moead_nd, ref_point)
            backend = 'handwritten'

        # HV difference: (DRL - MOEA/D) / MOEA/D * 100
        if hv_moead > 0:
            hv_diff_pct = (hv_drl - hv_moead) / hv_moead * 100
        else:
            hv_diff_pct = 0.0

        rows.append({
            'size': size,
            'drl_nd': len(drl_nd),
            'moead_nd': len(moead_nd),
            'hv_drl': hv_drl,
            'hv_moead': hv_moead,
            'hv_diff_pct': hv_diff_pct,
            'ref': ref_point,
        })

    # Print table
    print()
    print("=" * 70)
    print("Hypervolume Comparison: DRL vs MOEA/D")
    print("=" * 70)
    exp_50 = opts.drl_exp_50 or opts.drl_exp
    exp_100 = opts.drl_exp_100 or opts.drl_exp
    print(f"DRL n=50:   paretofront/pareto_results_50_{exp_50}.pkl")
    print(f"DRL n=100:  paretofront/pareto_results_100_{exp_100}.pkl")
    print(f"MOEA/D:     paretofront/moead_results_{{50,100}}_exp1.pkl")
    print(f"Reference:   (max_f1 * 1.1, max_f2 * 1.1) over all points")
    print("=" * 70)
    print(f"{'n':>4} | {'DRL_nd':>8} | {'MOEA/D_nd':>10} | {'HV_DRL':>12} | {'HV_MOEA/D':>12} | {'HV_diff_%':>10}")
    print("-" * 70)
    for r in rows:
        print(f"{r['size']:>4} | {r['drl_nd']:>8} | {r['moead_nd']:>10} | "
              f"{r['hv_drl']:>12.2f} | {r['hv_moead']:>12.2f} | {r['hv_diff_pct']:>+9.2f}%")
    print("=" * 70)
    print("(nd = non-dominated count; HV_diff% = (DRL - MOEA/D) / MOEA/D * 100)")
    print()


if __name__ == '__main__':
    main()
