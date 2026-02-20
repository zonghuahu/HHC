# -*- coding: utf-8 -*-
"""
MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)
baseline for HHCRSP (Home Health Care Routing and Scheduling Problem).

Produces a Pareto front of (f1=total_distance, f2=total_waiting_time)
comparable to the DRL model for thesis comparison.

Evaluation replicates agh_baseline.py val() exactly:
  - Sequential fleet solving order: [1, 2, 3, 4, 5, 6]
  - Time-window propagation between precedence stages
  - Fleet 1: 0-min gap; others: +10 min gap
  - tw_right tightening by next_duration for stage-0 fleets
  - Depot visits reset cur_time to -60 (INIT_FREE_TIME)
"""

import os
import copy
import math
import time
import random
import pickle
import argparse
import numpy as np

# ============================================================================
# Constants
# ============================================================================
NODE_SIZE = 92        # 91 patient locations + 1 depot
SPEED = 80.0          # distance_units / minute
DEPOT_TW_RIGHT = 1441 # effectively unlimited
INIT_FREE_TIME = -60  # vehicle available 60 min before day starts

# Need → fleet mapping
FLEET_NEEDS = {
    1: [1, 9],
    2: [2, 7],
    3: [3, 7],
    4: [4, 8],
    5: [5, 8],
    6: [6, 9],
}

# ============================================================================
# Data Loading
# ============================================================================

def load_resources(base_dir='problems/agh'):
    """Load static resources: distance matrix, fleet info, arrival probs."""
    with open(os.path.join(base_dir, 'distance.pkl'), 'rb') as f:
        distance_dict = pickle.load(f)

    # Convert to 92x92 numpy array for O(1) lookup
    dist_matrix = np.zeros((NODE_SIZE, NODE_SIZE), dtype=np.float64)
    for (i, j), d in distance_dict.items():
        dist_matrix[i, j] = d

    with open(os.path.join(base_dir, 'fleet_info.pkl'), 'rb') as f:
        fleet_info = pickle.load(f)

    arrival_prob = np.load(os.path.join(base_dir, 'arrival_prob.npy'))

    return dist_matrix, fleet_info, arrival_prob


def generate_instances(n_instances, graph_size, arrival_prob, seed=None):
    """Generate instances matching AGHDataset's empirical distribution."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    n_hour = np.arange(10, 20)
    n_min = 60
    n_gate = 91

    instances = []
    for _ in range(n_instances):
        loc = 1 + rng.choice(n_gate, size=graph_size)
        arrival = (60 * rng.choice(n_hour, size=graph_size, p=arrival_prob)
                   + rng.randint(0, n_min, size=graph_size))
        type_ = rng.randint(0, 3, size=graph_size)
        departure = arrival + 120  # fixed 120-min window
        # Need: ~70% single (1-6), ~30% combo (7-9)
        need_vals = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        need_probs = np.array([0.116, 0.116, 0.116, 0.116, 0.116, 0.116,
                               0.1, 0.1, 0.1])
        need_probs = need_probs / need_probs.sum()
        need = rng.choice(need_vals, size=graph_size, p=need_probs)

        instances.append({
            'loc': loc.astype(np.int64),
            'arrival': arrival.astype(np.float64),
            'departure': departure.astype(np.float64),
            'type': type_.astype(np.int64),
            'need': need.astype(np.int64),
        })

    return instances


def load_instances_from_pkl(filename, n_instances=None, offset=0):
    """Load instances from a .pkl file (same format as AGHDataset)."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    if n_instances is not None:
        data = data[offset:offset + n_instances]

    instances = []
    for item in data:
        loc, arrival, departure, type_, need = item[:5]
        instances.append({
            'loc': np.array(loc, dtype=np.int64),
            'arrival': np.array(arrival, dtype=np.float64),
            'departure': np.array(departure, dtype=np.float64),
            'type': np.array(type_, dtype=np.int64),
            'need': np.array(need, dtype=np.int64),
        })

    return instances


# ============================================================================
# Fleet-patient assignment
# ============================================================================

def get_fleet_patients(need, fleet_id):
    """Get patient indices (0-based) assigned to a given fleet."""
    valid_needs = FLEET_NEEDS[fleet_id]
    return [i for i, n in enumerate(need) if n in valid_needs]


# ============================================================================
# Sequential Evaluation (exact replication of DRL pipeline)
# ============================================================================

def simulate_fleet_route(route, instance, dist_matrix, fleet_info, fleet_id,
                         tw_left_f, tw_right_f):
    """
    Simulate a single fleet's route.
    Route is a list of patient indices (0-based). Depot returns are implicit:
    the vehicle starts at depot, visits patients in order, and returns to depot.

    When the vehicle cannot reach the next patient within its tw, it returns
    to depot (reset cur_time=-60) and starts a new trip.

    Returns: (f1, f2, serve_times_dict)
    """
    loc = instance['loc']
    type_ = instance['type']
    fleet_dur = fleet_info['duration'][fleet_id]

    f1 = 0.0
    f2 = 0.0
    cur_time = INIT_FREE_TIME
    cur_loc = 0  # depot
    serve_times = {}

    for p in route:
        p_loc = loc[p]

        # Check if we can reach this patient from current position
        travel_dist = dist_matrix[cur_loc, p_loc]
        travel_time = travel_dist / SPEED
        arr_time = cur_time + travel_time
        svc_start = max(arr_time, tw_left_f[p])
        dur_p = fleet_dur[type_[p]]

        if svc_start + dur_p > tw_right_f[p] + 1e-5:
            # Cannot serve from current position; return to depot and retry
            if cur_loc != 0:
                f1 += dist_matrix[cur_loc, 0]  # return to depot distance
                cur_time = INIT_FREE_TIME
                cur_loc = 0

                # Retry from depot
                travel_dist = dist_matrix[0, p_loc]
                travel_time = travel_dist / SPEED
                arr_time = cur_time + travel_time
                svc_start = max(arr_time, tw_left_f[p])
                dur_p = fleet_dur[type_[p]]

        # Add distance
        f1 += dist_matrix[cur_loc, p_loc]

        # Waiting time = patient waits from tw_left to service start
        waiting = max(0.0, svc_start - tw_left_f[p])
        f2 += waiting

        serve_times[p] = svc_start
        cur_time = svc_start + dur_p
        cur_loc = p_loc

    # Return to depot
    if cur_loc != 0:
        f1 += dist_matrix[cur_loc, 0]

    return f1, f2, serve_times


def evaluate_solution(solution, instance, dist_matrix, fleet_info):
    """
    Evaluate a solution (dict: fleet_id -> list of patient indices).
    Returns (f1_total, f2_total).

    Replicates agh_baseline.py val() sequential evaluation:
      1. Process fleets in order [1,2,3,4,5,6]
      2. Each fleet reads tw_left from its precedence stage
      3. tw_right tightened by next_duration for the fleet's stage
      4. After each fleet, propagate serve_times to next stage
    """
    graph_size = len(instance['loc'])
    type_ = instance['type']
    need = instance['need']
    departure = instance['departure']
 
    n_stages = len(fleet_info['next_duration']) + 1
    tw_left_stages = np.tile(instance['arrival'], (n_stages, 1)).astype(np.float64)

    f1_total = 0.0
    f2_total = 0.0

    for f in fleet_info['order']:
        prec_stage = fleet_info['precedence'][f]

        # Time windows for this fleet
        tw_left_f = tw_left_stages[prec_stage].copy()
        next_dur = fleet_info['next_duration'][prec_stage]
        tw_right_f = np.array([departure[i] - next_dur[type_[i]]
                               for i in range(graph_size)])

        route = solution.get(f, [])

        # Simulate
        f1_f, f2_f, serve_times = simulate_fleet_route(
            route, instance, dist_matrix, fleet_info, f, tw_left_f, tw_right_f)

        f1_total += f1_f
        f2_total += f2_f

        # Propagate tw_left to next stage
        next_stage = prec_stage + 1
        if next_stage < n_stages:
            fleet_patients_set = set(get_fleet_patients(need, f))
            for p in fleet_patients_set:
                if p in serve_times:
                    gap = 0 if f == 1 else 10
                    tw_left_stages[next_stage, p] = serve_times[p] + gap

    return f1_total, f2_total


# ============================================================================
# Solution Construction
# ============================================================================

def construct_nn_solution(instance, dist_matrix, fleet_info):
    """
    Greedy nearest-neighbor construction per fleet, following sequential order.
    Inserts depot returns when no patient is feasible from current position.
    """
    graph_size = len(instance['loc'])
    loc = instance['loc']
    type_ = instance['type']
    need = instance['need']
    departure = instance['departure']

    n_stages = len(fleet_info['next_duration']) + 1
    tw_left_stages = np.tile(instance['arrival'], (n_stages, 1)).astype(np.float64)

    solution = {}

    for f in fleet_info['order']:
        prec_stage = fleet_info['precedence'][f]
        tw_left_f = tw_left_stages[prec_stage].copy()
        next_dur = fleet_info['next_duration'][prec_stage]
        tw_right_f = np.array([departure[i] - next_dur[type_[i]]
                               for i in range(graph_size)])
        fleet_dur = fleet_info['duration'][f]

        fleet_patients = get_fleet_patients(need, f)
        unvisited = set(fleet_patients)
        route = []

        cur_time = INIT_FREE_TIME
        cur_loc = 0  # depot
        serve_times = {}

        while unvisited:
            # Find nearest feasible patient
            best_p = None
            best_dist = float('inf')

            for p in unvisited:
                p_loc = loc[p]
                d = dist_matrix[cur_loc, p_loc]
                travel_time = d / SPEED
                arr_time = cur_time + travel_time
                svc_start = max(arr_time, tw_left_f[p])
                dur_p = fleet_dur[type_[p]]

                if svc_start + dur_p <= tw_right_f[p] + 1e-5:
                    if d < best_dist:
                        best_dist = d
                        best_p = p

            if best_p is None:
                # No feasible patient from current position
                if cur_loc != 0:
                    # Return to depot, reset time
                    cur_time = INIT_FREE_TIME
                    cur_loc = 0
                    continue
                else:
                    # Even from depot no one is feasible — try earliest-tw patient
                    earliest_p = None
                    earliest_tw = float('inf')
                    for p in unvisited:
                        if tw_left_f[p] < earliest_tw:
                            earliest_tw = tw_left_f[p]
                            earliest_p = p
                    if earliest_p is not None:
                        best_p = earliest_p
                    else:
                        break

            # Visit best patient
            route.append(best_p)
            unvisited.discard(best_p)
            p_loc = loc[best_p]
            travel_time = dist_matrix[cur_loc, p_loc] / SPEED
            arr_time = cur_time + travel_time
            svc_start = max(arr_time, tw_left_f[best_p])
            serve_times[best_p] = svc_start
            cur_time = svc_start + fleet_dur[type_[best_p]]
            cur_loc = p_loc

        solution[f] = route

        # Propagate tw_left to next stage
        next_stage = prec_stage + 1
        if next_stage < n_stages:
            for p in fleet_patients:
                if p in serve_times:
                    gap = 0 if f == 1 else 10
                    tw_left_stages[next_stage, p] = serve_times[p] + gap

    return solution


def construct_nn_solution_sorted(instance, dist_matrix, fleet_info):
    """
    Sorted nearest-neighbor: process patients in tw_left order,
    return to depot when needed. Generally produces better initial solutions.
    """
    graph_size = len(instance['loc'])
    loc = instance['loc']
    type_ = instance['type']
    need = instance['need']
    departure = instance['departure']

    n_stages = len(fleet_info['next_duration']) + 1
    tw_left_stages = np.tile(instance['arrival'], (n_stages, 1)).astype(np.float64)

    solution = {}

    for f in fleet_info['order']:
        prec_stage = fleet_info['precedence'][f]
        tw_left_f = tw_left_stages[prec_stage].copy()
        next_dur = fleet_info['next_duration'][prec_stage]
        tw_right_f = np.array([departure[i] - next_dur[type_[i]]
                               for i in range(graph_size)])
        fleet_dur = fleet_info['duration'][f]

        fleet_patients = get_fleet_patients(need, f)
        # Sort by tw_left (earliest first)
        fleet_patients_sorted = sorted(fleet_patients, key=lambda p: tw_left_f[p])

        route = []
        cur_time = INIT_FREE_TIME
        cur_loc = 0
        serve_times = {}
        unvisited = list(fleet_patients_sorted)
        max_retries = len(unvisited) * 2 + 5
        retries = 0

        while unvisited and retries < max_retries:
            # Try to find a feasible patient
            best_p = None
            best_score = float('inf')

            for p in unvisited:
                p_loc = loc[p]
                d = dist_matrix[cur_loc, p_loc]
                travel_time = d / SPEED
                arr_time = cur_time + travel_time
                svc_start = max(arr_time, tw_left_f[p])
                dur_p = fleet_dur[type_[p]]

                if svc_start + dur_p <= tw_right_f[p] + 1e-5:
                    # Score: prioritize by tw_left, break ties by distance
                    score = tw_left_f[p] + d * 0.01
                    if score < best_score:
                        best_score = score
                        best_p = p

            if best_p is None:
                if cur_loc != 0:
                    cur_time = INIT_FREE_TIME
                    cur_loc = 0
                    retries += 1
                    continue
                else:
                    # Force the earliest tw_left patient
                    best_p = unvisited[0]
                    retries += 1

            route.append(best_p)
            unvisited.remove(best_p)
            p_loc = loc[best_p]
            travel_time = dist_matrix[cur_loc, p_loc] / SPEED
            arr_time = cur_time + travel_time
            svc_start = max(arr_time, tw_left_f[best_p])
            serve_times[best_p] = svc_start
            cur_time = svc_start + fleet_dur[type_[best_p]]
            cur_loc = p_loc
            retries = 0

        solution[f] = route

        next_stage = prec_stage + 1
        if next_stage < n_stages:
            for p in fleet_patients:
                if p in serve_times:
                    gap = 0 if f == 1 else 10
                    tw_left_stages[next_stage, p] = serve_times[p] + gap

    return solution


def construct_random_solution(instance, dist_matrix, fleet_info):
    """Random construction: shuffle patients per fleet, greedily insert with depot returns."""
    graph_size = len(instance['loc'])
    loc = instance['loc']
    type_ = instance['type']
    need = instance['need']
    departure = instance['departure']

    n_stages = len(fleet_info['next_duration']) + 1
    tw_left_stages = np.tile(instance['arrival'], (n_stages, 1)).astype(np.float64)

    solution = {}

    for f in fleet_info['order']:
        prec_stage = fleet_info['precedence'][f]
        tw_left_f = tw_left_stages[prec_stage].copy()
        next_dur = fleet_info['next_duration'][prec_stage]
        tw_right_f = np.array([departure[i] - next_dur[type_[i]]
                               for i in range(graph_size)])
        fleet_dur = fleet_info['duration'][f]

        fleet_patients = get_fleet_patients(need, f)
        random.shuffle(fleet_patients)

        route = []
        cur_time = INIT_FREE_TIME
        cur_loc = 0
        serve_times = {}
        unvisited = list(fleet_patients)
        max_retries = len(unvisited) * 3 + 5
        retries = 0

        while unvisited and retries < max_retries:
            p = unvisited[0]
            p_loc = loc[p]
            travel_time = dist_matrix[cur_loc, p_loc] / SPEED
            arr_time = cur_time + travel_time
            svc_start = max(arr_time, tw_left_f[p])
            dur_p = fleet_dur[type_[p]]

            if svc_start + dur_p <= tw_right_f[p] + 1e-5:
                route.append(p)
                unvisited.pop(0)
                serve_times[p] = svc_start
                cur_time = svc_start + dur_p
                cur_loc = p_loc
                retries = 0
            else:
                # Try depot return or rotate
                if cur_loc != 0:
                    cur_time = INIT_FREE_TIME
                    cur_loc = 0
                    retries += 1
                else:
                    # Rotate: move this patient to the end
                    unvisited.append(unvisited.pop(0))
                    retries += 1

        # Force-add any remaining
        for p in unvisited:
            route.append(p)
            p_loc = loc[p]
            travel_time = dist_matrix[cur_loc, p_loc] / SPEED
            arr_time = cur_time + travel_time
            svc_start = max(arr_time, tw_left_f[p])
            serve_times[p] = svc_start
            cur_time = svc_start + fleet_dur[type_[p]]
            cur_loc = p_loc

        solution[f] = route

        next_stage = prec_stage + 1
        if next_stage < n_stages:
            for p in fleet_patients:
                if p in serve_times:
                    gap = 0 if f == 1 else 10
                    tw_left_stages[next_stage, p] = serve_times[p] + gap

    return solution


# ============================================================================
# Genetic Operators
# ============================================================================

def crossover_ox(route1, route2):
    """Order Crossover (OX) between two routes of the same fleet.
    Both routes must contain exactly the same set of patient indices."""
    if len(route1) < 3:
        return route1[:], route2[:]

    size = len(route1)
    start = random.randint(0, size - 2)
    end = random.randint(start + 1, size - 1)

    # Child 1
    child1 = [None] * size
    child1[start:end + 1] = route1[start:end + 1]
    segment_set = set(route1[start:end + 1])
    fill = [g for g in route2 if g not in segment_set]
    idx = 0
    for i in range(size):
        if child1[i] is None:
            child1[i] = fill[idx]
            idx += 1

    # Child 2
    child2 = [None] * size
    child2[start:end + 1] = route2[start:end + 1]
    segment_set2 = set(route2[start:end + 1])
    fill2 = [g for g in route1 if g not in segment_set2]
    idx = 0
    for i in range(size):
        if child2[i] is None:
            child2[i] = fill2[idx]
            idx += 1

    return child1, child2


def mutate_relocate(route):
    """Move a random patient to a random position."""
    if len(route) < 2:
        return route[:]
    route = route[:]
    i = random.randint(0, len(route) - 1)
    p = route.pop(i)
    j = random.randint(0, len(route))
    route.insert(j, p)
    return route


def mutate_swap(route):
    """Swap two random patients."""
    if len(route) < 2:
        return route[:]
    route = route[:]
    i, j = random.sample(range(len(route)), 2)
    route[i], route[j] = route[j], route[i]
    return route


def mutate_reverse_segment(route):
    """Reverse a random segment (2-opt style)."""
    if len(route) < 3:
        return route[:]
    route = route[:]
    i = random.randint(0, len(route) - 2)
    j = random.randint(i + 1, len(route) - 1)
    route[i:j+1] = route[i:j+1][::-1]
    return route


def route_distance(route, loc, dist_matrix):
    """Calculate total distance of a single fleet route (depot→patients→depot)."""
    if not route:
        return 0.0
    total = 0.0
    cur_loc = 0
    for p in route:
        p_loc = loc[p]
        total += dist_matrix[cur_loc, p_loc]
        cur_loc = p_loc
    total += dist_matrix[cur_loc, 0]
    return total


# ============================================================================
# MOEA/D Framework
# ============================================================================

def generate_weight_vectors(n_weights):
    """Generate uniformly spaced weight vectors for 2 objectives."""
    weights = []
    for i in range(n_weights):
        w1 = i / (n_weights - 1)
        w2 = 1.0 - w1
        weights.append((w1, w2))
    return weights


def compute_neighborhoods(weights, T):
    """Compute T nearest neighbors for each weight vector."""
    n = len(weights)
    neighborhoods = []
    for i in range(n):
        dists = []
        for j in range(n):
            d = math.sqrt((weights[i][0] - weights[j][0])**2 +
                          (weights[i][1] - weights[j][1])**2)
            dists.append((d, j))
        dists.sort()
        neighbors = [idx for _, idx in dists[:T+1]]
        neighborhoods.append(neighbors)
    return neighborhoods


def tchebycheff(f1, f2, w, z_star):
    """Tchebycheff decomposition scalar value."""
    w1 = max(w[0], 1e-4)
    w2 = max(w[1], 1e-4)
    return max(w1 * abs(f1 - z_star[0]), w2 * abs(f2 - z_star[1]))


def apply_mutation(solution, instance, mutation_rate=0.3):
    """Apply mutation operators to a solution."""
    offspring = {}
    need = instance['need']

    for f in range(1, 7):
        route = solution[f][:]
        r = random.random()
        if r < mutation_rate / 3:
            route = mutate_swap(route)
        elif r < 2 * mutation_rate / 3:
            route = mutate_relocate(route)
        elif r < mutation_rate:
            route = mutate_reverse_segment(route)
        offspring[f] = route

    return offspring


def crossover_solutions(parent1, parent2, instance):
    """Crossover two solutions by applying OX per fleet."""
    child1, child2 = {}, {}
    need = instance['need']

    for f in range(1, 7):
        r1 = parent1[f]
        r2 = parent2[f]

        if set(r1) == set(r2) and len(r1) >= 3:
            c1, c2 = crossover_ox(r1, r2)
            child1[f] = c1
            child2[f] = c2
        else:
            child1[f] = r1[:]
            child2[f] = r2[:]

    return child1, child2


def moead(instances, dist_matrix, fleet_info, n_weights=11,
          n_gen=500, T=3, mutation_rate=0.3, verbose=True):
    """
    MOEA/D main algorithm.

    For each instance, evolves {n_weights} solutions (one per weight vector)
    using Tchebycheff decomposition. Collects results across all instances.
    """
    weights = generate_weight_vectors(n_weights)
    neighborhoods = compute_neighborhoods(weights, T)

    if verbose:
        print(f"MOEA/D: {n_weights} weight vectors, n_gen={n_gen}, T={T}")
        print(f"Evaluating on {len(instances)} instances")

    all_results = {i: {'f1': [], 'f2': []} for i in range(n_weights)}

    for inst_idx, instance in enumerate(instances):
        if verbose and ((inst_idx + 1) % 10 == 0 or inst_idx == 0):
            print(f"  Instance {inst_idx + 1}/{len(instances)}")

        # Initialize: one solution per weight vector
        population = []
        obj_values = []

        # First solution: sorted NN (typically best initial)
        sol0 = construct_nn_solution_sorted(instance, dist_matrix, fleet_info)
        f1_0, f2_0 = evaluate_solution(sol0, instance, dist_matrix, fleet_info)
        population.append(sol0)
        obj_values.append((f1_0, f2_0))

        # Additional solutions: mix of NN and random
        for w_idx in range(1, n_weights):
            if w_idx % 2 == 0:
                sol = construct_nn_solution(instance, dist_matrix, fleet_info)
            else:
                sol = construct_random_solution(instance, dist_matrix, fleet_info)
            f1, f2 = evaluate_solution(sol, instance, dist_matrix, fleet_info)
            population.append(sol)
            obj_values.append((f1, f2))

        # Reference point (ideal)
        z_star = [min(o[0] for o in obj_values), min(o[1] for o in obj_values)]

        # Evolution
        for gen in range(n_gen):
            for i in range(n_weights):
                neighbors = neighborhoods[i]

                # Select two parents from neighborhood
                p1_idx = random.choice(neighbors)
                p2_idx = random.choice(neighbors)
                while p2_idx == p1_idx and len(neighbors) > 1:
                    p2_idx = random.choice(neighbors)

                # Crossover
                child1, child2 = crossover_solutions(
                    population[p1_idx], population[p2_idx], instance)

                # Pick one child
                child = child1 if random.random() < 0.5 else child2

                # Mutation
                child = apply_mutation(child, instance, mutation_rate)

                # Evaluate
                f1_c, f2_c = evaluate_solution(child, instance,
                                               dist_matrix, fleet_info)

                # Update reference point
                z_star[0] = min(z_star[0], f1_c)
                z_star[1] = min(z_star[1], f2_c)

                # Update neighboring solutions via Tchebycheff
                for j in neighbors:
                    w = weights[j]
                    old_te = tchebycheff(obj_values[j][0], obj_values[j][1],
                                         w, z_star)
                    new_te = tchebycheff(f1_c, f2_c, w, z_star)

                    if new_te < old_te:
                        population[j] = copy.deepcopy(child)
                        obj_values[j] = (f1_c, f2_c)

        # Collect results
        for w_idx in range(n_weights):
            f1, f2 = obj_values[w_idx]
            all_results[w_idx]['f1'].append(f1)
            all_results[w_idx]['f2'].append(f2)

    # Compute statistics
    pareto_results = []
    for w_idx in range(n_weights):
        f1_vals = np.array(all_results[w_idx]['f1'])
        f2_vals = np.array(all_results[w_idx]['f2'])
        pareto_results.append({
            'lambda': list(weights[w_idx]),
            'f1_mean': float(np.mean(f1_vals)),
            'f1_std': float(np.std(f1_vals)),
            'f2_mean': float(np.mean(f2_vals)),
            'f2_std': float(np.std(f2_vals)),
        })

    return pareto_results, weights


# ============================================================================
# Smoke Test
# ============================================================================

def smoke_test():
    """Quick sanity check: verify evaluation produces reasonable costs."""
    print("=" * 60)
    print("Running smoke test...")
    print("=" * 60)
    dist_matrix, fleet_info, arrival_prob = load_resources()

    instances = generate_instances(1, 50, arrival_prob, seed=42)
    instance = instances[0]

    print(f"\nInstance: {len(instance['loc'])} patients")
    needs, counts = np.unique(instance['need'], return_counts=True)
    print(f"Need distribution: {dict(zip(needs.tolist(), counts.tolist()))}")

    # Test sorted NN construction
    sol = construct_nn_solution_sorted(instance, dist_matrix, fleet_info)

    for f in range(1, 7):
        print(f"  Fleet {f}: {len(sol[f])} patients")

    f1, f2 = evaluate_solution(sol, instance, dist_matrix, fleet_info)
    print(f"\nSorted NN: f1={f1:.2f}, f2={f2:.2f}")

    # Test regular NN
    sol2 = construct_nn_solution(instance, dist_matrix, fleet_info)
    f1_2, f2_2 = evaluate_solution(sol2, instance, dist_matrix, fleet_info)
    print(f"Regular NN: f1={f1_2:.2f}, f2={f2_2:.2f}")

    # Test random construction
    sol3 = construct_random_solution(instance, dist_matrix, fleet_info)
    f1_3, f2_3 = evaluate_solution(sol3, instance, dist_matrix, fleet_info)
    print(f"Random:     f1={f1_3:.2f}, f2={f2_3:.2f}")

    # Sanity checks
    # DRL n=50 results: f1 ≈ 2595-3003, f2 ≈ 3-211
    # Baselines should be somewhat worse, but same order of magnitude
    assert 500 < f1 < 15000, f"f1={f1} out of expected range"
    assert f2 >= 0, f"f2={f2} should be non-negative"

    print("\n--- Mini MOEA/D run (3 instances, 20 gen) ---")
    mini_instances = generate_instances(3, 50, arrival_prob, seed=42)
    results, weights = moead(mini_instances, dist_matrix, fleet_info,
                             n_weights=5, n_gen=20, T=2,
                             mutation_rate=0.3, verbose=False)
    for r in results:
        print(f"  λ={r['lambda']} → f1={r['f1_mean']:.1f}, f2={r['f2_mean']:.1f}")

    print("\nSmoke test PASSED ✓")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='MOEA/D baseline for HHCRSP')
    parser.add_argument('--graph_size', type=int, default=50,
                        help='Problem size (50 or 100)')
    parser.add_argument('--n_instances', type=int, default=100,
                        help='Number of test instances')
    parser.add_argument('--filename', type=str, default=None,
                        help='Load instances from .pkl file')
    parser.add_argument('--n_weights', type=int, default=11,
                        help='Number of weight vectors (Pareto points)')
    parser.add_argument('--n_gen', type=int, default=500,
                        help='Number of generations')
    parser.add_argument('--T', type=int, default=3,
                        help='Neighborhood size')
    parser.add_argument('--mutation_rate', type=float, default=0.3,
                        help='Mutation probability')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='paretofront',
                        help='Output directory')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Run smoke test and exit')
    parser.add_argument('--no_progress', action='store_true',
                        help='Suppress progress output')

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
        print(f"Generating {opts.n_instances} instances "
              f"(n={opts.graph_size}, seed={opts.seed})")
        instances = generate_instances(opts.n_instances, opts.graph_size,
                                       arrival_prob, seed=opts.seed)

    start_time = time.time()
    pareto_results, weights = moead(
        instances, dist_matrix, fleet_info,
        n_weights=opts.n_weights,
        n_gen=opts.n_gen,
        T=opts.T,
        mutation_rate=opts.mutation_rate,
        verbose=not opts.no_progress,
    )
    elapsed = time.time() - start_time

    # Print results table
    print("\n" + "=" * 70)
    print(f"  MOEA/D Results | n={opts.graph_size} | "
          f"{opts.n_instances} instances | {elapsed:.1f}s")
    print(f"{'':>4}{'λ₁':>6}{'λ₂':>6}"
          f"{'f₁ (Distance)':>20}{'f₂ (Waiting)':>20}")
    print("-" * 70)
    for r in pareto_results:
        l1, l2 = r['lambda']
        print(f"  {l1:5.1f}  {l2:5.1f}"
              f"  {r['f1_mean']:8.2f} ± {r['f1_std']:<8.2f}"
              f"  {r['f2_mean']:8.2f} ± {r['f2_std']:<8.2f}")
    print("=" * 70)

    # Save results (same format as DRL pareto_results_*.pkl)
    os.makedirs(opts.output_dir, exist_ok=True)
    out_path = os.path.join(opts.output_dir,
                            f'moead_results_{opts.graph_size}_exp1.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(pareto_results, f)
    print(f"\nResults saved to {out_path}")

    txt_path = os.path.join(opts.output_dir,
                            f'moead_results_{opts.graph_size}_exp1.txt')
    with open(txt_path, 'w') as f:
        f.write(f"MOEA/D Results (n={opts.graph_size}, "
                f"{opts.n_instances} instances)\n")
        f.write(f"Gen={opts.n_gen}, T={opts.T}, "
                f"Mutation={opts.mutation_rate}, Seed={opts.seed}\n")
        f.write(f"Time: {elapsed:.1f}s\n\n")
        f.write(f"{'l1':>6}{'l2':>6}{'f1_mean':>12}{'f1_std':>10}"
                f"{'f2_mean':>12}{'f2_std':>10}\n")
        for r in pareto_results:
            l1, l2 = r['lambda']
            f.write(f"{l1:6.1f}{l2:6.1f}{r['f1_mean']:12.2f}"
                    f"{r['f1_std']:10.2f}{r['f2_mean']:12.2f}"
                    f"{r['f2_std']:10.2f}\n")
    print(f"Text results saved to {txt_path}")


if __name__ == '__main__':
    main()
