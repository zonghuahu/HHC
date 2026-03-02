# -*- coding: utf-8 -*-
"""
hhc_problem_pymoo.py

Wraps the HHCRSP evaluation from moead.py into a pymoo ElementwiseProblem,
enabling the use of pymoo's NSGA2, NSGA3, MOEA/D, etc. on the HHC problem.

Encoding (Method A – global permutation):
  x is a permutation of [0, graph_size-1] (all patient indices).
  For each fleet f, the relative order of patients belonging to f in x
  defines that fleet's route. This allows standard OX crossover and
  keeps n_var = graph_size regardless of need distribution.

Objectives:
  f1 = total travel distance  (same as moead.py)
  f2 = total patient waiting time  (same as moead.py)

Usage examples:
  python hhc_problem_pymoo.py --smoke_test
  python hhc_problem_pymoo.py --algorithm NSGA2 --n_instances 10 --n_gen 200
  python hhc_problem_pymoo.py --algorithm MOEAD --n_instances 10 --n_gen 200
"""

import os
import copy
import json
import time
import random
import argparse
import pickle
import numpy as np

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# Import all HHC-specific logic from moead.py (no modification needed there)
from moead import (
    load_resources,
    generate_instances,
    load_instances_from_pkl,
    get_fleet_patients,
    evaluate_solution,
    crossover_ox,
    mutate_swap,
    mutate_relocate,
    mutate_reverse_segment,
    construct_nn_solution_sorted,
    construct_nn_solution,
    construct_random_solution,
    FLEET_NEEDS,
)


# ============================================================================
# HHC Problem Definition
# ============================================================================

class HHCProblem(ElementwiseProblem):
    """
    HHCRSP as a pymoo ElementwiseProblem.

    Decision variable x: a permutation of [0, graph_size-1].
    Per-fleet route = relative order of that fleet's patients in x.
    """

    def __init__(self, instance, dist_matrix, fleet_info, **kwargs):
        self.instance = instance
        self.dist_matrix = dist_matrix
        self.fleet_info = fleet_info

        graph_size = len(instance['loc'])
        # Precompute which patients belong to each fleet
        self.fleet_patients = {
            f: get_fleet_patients(instance['need'], f)
            for f in range(1, 7)
        }

        super().__init__(
            n_var=graph_size,
            n_obj=2,
            xl=0,
            xu=graph_size - 1,
            **kwargs
        )

    def _decode(self, x):
        """
        Decode a global permutation into {fleet_id: [patient_indices]}.

        For each fleet, extract its patients from x in the order they appear.
        This preserves the relative ordering encoded by genetic operators.
        """
        # Build position lookup: patient_idx -> position in x
        pos = {int(p): i for i, p in enumerate(x)}

        solution = {}
        for f, patients in self.fleet_patients.items():
            if patients:
                # Sort patients by their position in the global permutation
                route = sorted(patients, key=lambda p: pos.get(p, float('inf')))
            else:
                route = []
            solution[f] = route
        return solution

    def _evaluate(self, x, out, *args, **kwargs):
        solution = self._decode(x)
        f1, f2 = evaluate_solution(
            solution, self.instance, self.dist_matrix, self.fleet_info
        )
        out['F'] = [f1, f2]


# ============================================================================
# Solution ↔ Permutation Conversion
# ============================================================================

def solution_to_permutation(solution, n):
    """
    Convert a solution dict {fleet_id: [patient_list]} to a global permutation.

    Patients are placed in the order they appear across fleets (fleet 1 first,
    then fleet 2, etc.). Patients already placed by an earlier fleet (combo-need
    overlap) are skipped. Any remaining patients are appended at the end.
    """
    perm = []
    seen = set()
    for f in sorted(solution.keys()):
        for p in solution[f]:
            if p not in seen:
                perm.append(p)
                seen.add(p)
    # Append any patients not covered (safety)
    for p in range(n):
        if p not in seen:
            perm.append(p)
    return perm


# ============================================================================
# Custom Sampling (with heuristic initialization)
# ============================================================================

class HHCSampling(Sampling):
    """
    Generate initial population using moead.py's heuristic constructors for
    the first 3 individuals (sorted NN, regular NN, random construction),
    then fill the rest with random permutations for diversity.
    """

    def _do(self, problem, n_samples, **kwargs):
        n = problem.n_var
        instance = problem.instance
        dist_matrix = problem.dist_matrix
        fleet_info = problem.fleet_info

        X = np.zeros((n_samples, n), dtype=int)

        # Heuristic constructors: sorted NN, regular NN, random construction
        heuristics = [
            construct_nn_solution_sorted,
            construct_nn_solution,
            construct_random_solution,
        ]

        for i in range(n_samples):
            if i < len(heuristics):
                sol = heuristics[i](instance, dist_matrix, fleet_info)
                perm = solution_to_permutation(sol, n)
            else:
                perm = np.random.permutation(n).tolist()
            X[i] = perm

        return X


# ============================================================================
# Custom Crossover: per-fleet OX on sub-sequences
# ============================================================================

class HHCCrossover(Crossover):
    """
    Order Crossover applied independently to each fleet's sub-sequence.

    For a pair of parents (p1, p2):
      - For each fleet f, extract the subsequence of that fleet's patients
        from each parent, apply OX, and recombine into a new global permutation.
    """

    def __init__(self, **kwargs):
        # n_parents=2, n_offsprings=2
        super().__init__(2, 2, **kwargs)

    def _do(self, problem, X, **kwargs):
        n_matings = X.shape[1]
        Y = np.full_like(X, fill_value=-1)

        for k in range(n_matings):
            p1 = list(X[0, k])
            p2 = list(X[1, k])

            c1, c2 = self._crossover_permutations(p1, p2, problem.fleet_patients)
            Y[0, k] = c1
            Y[1, k] = c2

        return Y

    @staticmethod
    def _crossover_permutations(p1, p2, fleet_patients):
        """
        Apply OX independently to each fleet's sub-sequence within the global
        permutation. The cross-fleet relative order is inherited from parent1 /
        parent2 respectively, which keeps len(child) == len(parent) == n_var.

        For each fleet f:
          - Extract the positions held by f's patients in parent1 and parent2.
          - Apply OX on the two orderings of those patients.
          - Write the OX result back into the *same positions* that f's patients
            occupied in each parent (so total length never changes).
        """
        c1 = p1[:]
        c2 = p2[:]

        # position maps for each parent
        pos1 = {v: i for i, v in enumerate(p1)}
        pos2 = {v: i for i, v in enumerate(p2)}

        # Track which global positions have already been reassigned
        # (patients can be shared across fleets due to combo-need, so we
        # only allow each patient to be touched once – first fleet wins)
        touched1 = set()
        touched2 = set()

        for f in sorted(fleet_patients.keys()):
            patients = fleet_patients[f]
            if len(patients) < 2:
                continue

            # Patients not yet reassigned for this child
            avail1 = [p for p in patients if p not in touched1]
            avail2 = [p for p in patients if p not in touched2]

            if len(avail1) < 2 or len(avail2) < 2:
                touched1.update(avail1)
                touched2.update(avail2)
                continue

            # Sub-sequences: ordering of avail patients in each parent
            subseq1 = sorted(avail1, key=lambda p: pos1[p])
            subseq2 = sorted(avail2, key=lambda p: pos2[p])

            # OX on sub-sequences
            ox1, ox2 = crossover_ox(subseq1, subseq2)

            # The global positions these patients held in each parent
            slots1 = sorted([pos1[p] for p in avail1])
            slots2 = sorted([pos2[p] for p in avail2])

            # Write OX result back into those slots (preserving position set)
            for slot, patient in zip(slots1, ox1):
                c1[slot] = patient
            for slot, patient in zip(slots2, ox2):
                c2[slot] = patient

            touched1.update(avail1)
            touched2.update(avail2)

        return c1, c2


# ============================================================================
# Custom Mutation
# ============================================================================

class HHCMutation(Mutation):
    """
    Mutation applied to the global permutation by operating on one fleet's
    sub-sequence at a time (swap, relocate, or reverse-segment).
    """

    def __init__(self, prob_mutation=0.3, **kwargs):
        super().__init__(**kwargs)
        self.prob_mutation = prob_mutation

    def _do(self, problem, X, **kwargs):
        Y = X.copy()

        for i in range(len(X)):
            if np.random.random() > self.prob_mutation:
                continue

            x = Y[i].tolist()
            pos = {v: j for j, v in enumerate(x)}

            # Pick a random fleet and mutate its sub-sequence
            f = random.randint(1, 6)
            patients = problem.fleet_patients.get(f, [])
            if len(patients) < 2:
                continue

            subseq = sorted(patients, key=lambda p: pos[p])

            r = random.random()
            if r < 1 / 3:
                new_subseq = mutate_swap(subseq)
            elif r < 2 / 3:
                new_subseq = mutate_relocate(subseq)
            else:
                new_subseq = mutate_reverse_segment(subseq)

            # Reconstruct x: replace positions of this fleet's patients
            for old_p, new_p in zip(subseq, new_subseq):
                old_pos = pos[old_p]
                x[old_pos] = new_p

            Y[i] = x

        return Y


# ============================================================================
# Run a single instance with pymoo algorithm
# ============================================================================

def run_instance(instance, dist_matrix, fleet_info, algorithm, n_gen):
    """Run pymoo optimizer on a single HHC instance. Returns res.F (Pareto front)."""
    problem = HHCProblem(instance, dist_matrix, fleet_info)
    termination = get_termination("n_gen", n_gen)

    res = minimize(
        problem,
        algorithm,
        termination,
        verbose=False,
        seed=random.randint(0, 10000),
    )

    return res.F  # shape (n_solutions, 2)


# ============================================================================
# Build algorithm
# ============================================================================

def build_algorithm(algo_name, pop_size=100):
    sampling = HHCSampling()
    crossover = HHCCrossover()
    mutation = HHCMutation(prob_mutation=0.3)

    if algo_name == "NSGA2":
        return NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=False,
        )
    elif algo_name == "NSGA3":
        ref_dirs = get_reference_directions("uniform", 2, n_partitions=pop_size - 1)
        return NSGA3(
            ref_dirs=ref_dirs,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=False,
        )
    elif algo_name == "MOEAD":
        ref_dirs = get_reference_directions("uniform", 2, n_partitions=pop_size - 1)
        return MOEAD(
            ref_dirs,
            n_neighbors=15,
            prob_neighbor_mating=0.7,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}. Use NSGA2, NSGA3, or MOEAD.")


# ============================================================================
# Smoke Test
# ============================================================================

def smoke_test():
    print("=" * 60)
    print("Smoke test: HHCProblem (pymoo wrapper)")
    print("=" * 60)

    dist_matrix, fleet_info, arrival_prob = load_resources()
    instances = generate_instances(1, 50, arrival_prob, seed=42)
    instance = instances[0]

    problem = HHCProblem(instance, dist_matrix, fleet_info)
    print(f"\nn_var={problem.n_var}, n_obj={problem.n_obj}")
    for f, pts in problem.fleet_patients.items():
        print(f"  Fleet {f}: {len(pts)} patients")

    # Test decode
    x = np.random.permutation(problem.n_var)
    solution = problem._decode(x)
    f1_pymoo, f2_pymoo = [None, None]
    out = {}
    problem._evaluate(x, out)
    f1_pymoo, f2_pymoo = out['F']

    # Cross-check with direct evaluate_solution
    f1_direct, f2_direct = evaluate_solution(solution, instance, dist_matrix, fleet_info)

    print(f"\n_evaluate:         f1={f1_pymoo:.2f}, f2={f2_pymoo:.2f}")
    print(f"evaluate_solution: f1={f1_direct:.2f}, f2={f2_direct:.2f}")
    assert abs(f1_pymoo - f1_direct) < 1e-6, "f1 mismatch!"
    assert abs(f2_pymoo - f2_direct) < 1e-6, "f2 mismatch!"
    print("✓ _evaluate matches evaluate_solution")

    # Test sampling
    sampler = HHCSampling()
    X_sample = sampler._do(problem, 5)
    assert X_sample.shape == (5, problem.n_var)
    print("✓ HHCSampling OK")

    # Mini NSGA2 run (3 gen, small pop)
    print("\nMini NSGA2 run (pop=10, gen=5)...")
    algo = build_algorithm("NSGA2", pop_size=10)
    F = run_instance(instance, dist_matrix, fleet_info, algo, n_gen=5)
    print(f"  Pareto front size: {len(F)}")
    print(f"  f1 range: {F[:,0].min():.1f} – {F[:,0].max():.1f}")
    print(f"  f2 range: {F[:,1].min():.1f} – {F[:,1].max():.1f}")

    print("\nSmoke test PASSED ✓")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HHC problem via pymoo (NSGA2 / NSGA3 / MOEA/D)")
    parser.add_argument('--algorithm', default='NSGA2',
                        help='Algorithm: NSGA2, NSGA3, MOEAD')
    parser.add_argument('--graph_size', type=int, default=50)
    parser.add_argument('--n_instances', type=int, default=10)
    parser.add_argument('--n_gen', type=int, default=500)
    parser.add_argument('--pop_size', type=int, default=100)
    parser.add_argument('--filename', type=str, default=None,
                        help='Load instances from .pkl file')
    parser.add_argument('--output_dir', type=str, default='paretofront_pymoo')
    parser.add_argument('--moead_format', action='store_true',
                        help='Output pareto_results format (lambda, f1_mean, f2_mean) for MOEA/D vs DRL comparison')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--smoke_test', action='store_true')
    opts = parser.parse_args()

    random.seed(opts.seed)
    np.random.seed(opts.seed)

    if opts.smoke_test:
        smoke_test()
        return

    dist_matrix, fleet_info, arrival_prob = load_resources()

    if opts.filename:
        print(f"Loading instances from {opts.filename}")
        instances = load_instances_from_pkl(opts.filename, opts.n_instances)
    else:
        print(f"Generating {opts.n_instances} instances (n={opts.graph_size}, seed={opts.seed})")
        instances = generate_instances(opts.n_instances, opts.graph_size,
                                       arrival_prob, seed=opts.seed)

    algorithm = build_algorithm(opts.algorithm, pop_size=opts.pop_size)
    os.makedirs(opts.output_dir, exist_ok=True)

    print(f"\nRunning {opts.algorithm} | pop={opts.pop_size} | gen={opts.n_gen}")
    print(f"{'Instance':<12} {'|F|':<6} {'f1_min':>10} {'f2_min':>10}  time(s)")
    print("-" * 55)

    all_fronts = []
    total_start = time.time()

    for idx, instance in enumerate(instances):
        t0 = time.time()
        algo = build_algorithm(opts.algorithm, pop_size=opts.pop_size)
        F = run_instance(instance, dist_matrix, fleet_info, algo, opts.n_gen)
        elapsed = time.time() - t0
        print(f"{idx+1:<12} {len(F):<6} {F[:,0].min():>10.2f} {F[:,1].min():>10.2f}  {elapsed:.1f}")
        all_fronts.append(F)

    total_elapsed = time.time() - total_start
    h, r = divmod(int(total_elapsed), 3600)
    m, s = divmod(r, 60)
    print("\n" + "=" * 60)
    print("pymoo {} completed. Total runtime: {:d}h {:02d}m {:02d}s ({:.1f}s)".format(
        opts.algorithm, h, m, s, total_elapsed))
    print("=" * 60)

    # Save
    if opts.moead_format and opts.algorithm == "MOEAD":
        # Aggregate into pareto_results format (same as moead.py) for DRL comparison
        ref_dirs = get_reference_directions("uniform", 2, n_partitions=opts.pop_size - 1)
        n_ref = len(ref_dirs)
        pareto_results = []
        for i in range(n_ref):
            f1_vals = np.array([F[i, 0] for F in all_fronts])
            f2_vals = np.array([F[i, 1] for F in all_fronts])
            l1, l2 = float(ref_dirs[i][0]), float(ref_dirs[i][1])
            pareto_results.append({
                'lambda': [l1, l2],
                'f1_mean': float(np.mean(f1_vals)),
                'f1_std': float(np.std(f1_vals)),
                'f2_mean': float(np.mean(f2_vals)),
                'f2_std': float(np.std(f2_vals)),
            })
        # Sort by lambda[0] to match moead order (0 -> 1)
        pareto_results.sort(key=lambda r: r['lambda'][0])
        out_pkl = os.path.join(opts.output_dir,
                               f'pymoo_moead_results_{opts.graph_size}_exp1.pkl')
        out_txt = os.path.join(opts.output_dir,
                               f'pymoo_moead_results_{opts.graph_size}_exp1.txt')
        with open(out_pkl, 'wb') as fh:
            pickle.dump(pareto_results, fh)
        with open(out_txt, 'w') as fh:
            fh.write(f"{'l1':>6}{'l2':>6}{'f1_mean':>12}{'f1_std':>10}"
                     f"{'f2_mean':>12}{'f2_std':>10}\n")
            for r in pareto_results:
                l1, l2 = r['lambda']
                fh.write(f"{l1:6.1f}{l2:6.1f}{r['f1_mean']:12.2f}"
                         f"{r['f1_std']:10.2f}{r['f2_mean']:12.2f}"
                         f"{r['f2_std']:10.2f}\n")
        print(f"Saved pareto_results (moead format) to {out_pkl}")
    else:
        out_path = os.path.join(opts.output_dir,
                                f'pymoo_{opts.algorithm}_{opts.graph_size}.pkl')
        with open(out_path, 'wb') as fh:
            pickle.dump(all_fronts, fh)
        print(f"Saved Pareto fronts to {out_path}")


if __name__ == '__main__':
    main()
