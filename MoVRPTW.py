import os
from generate_test_dataset import load_dataset
import numpy as np
import argparse
import time
from pymoo.core.repair import Repair
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.operators.crossover.erx import ERX
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination import get_termination
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.problem import Problem
from scipy.spatial import distance
# from krodata_large_size.convet_benchmark_dataloader import load_data

class MoTravelingSalesman(ElementwiseProblem):

    def __init__(self, cities, **kwargs):
        n_cities, _ = cities.shape
        self.cities = cities
        coords1 = cities[:, :2]
        coords2 = cities[:, 2:]
        self.dist1 = np.linalg.norm(
                    coords1[None, :, :].repeat(n_cities, axis=0) - coords1[:, None, :].repeat(n_cities, axis=1),
                    axis=-1)
        self.dist2 = np.linalg.norm(
                    coords2[None, :, :].repeat(n_cities, axis=0) - coords2[:, None, :].repeat(n_cities, axis=1),
                    axis=-1)

        super(MoTravelingSalesman, self).__init__(
            n_var=n_cities,
            n_obj=2,
            xl=0,
            xu=n_cities - 1,
            type_var=int,
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        L = x
        C = x

        f1 = self.get_route_length1(L)
        f2 = self.get_route_length2(C)

        out['F'] = np.column_stack([f1, f2])

    def get_route_length1(self, x):
        n_cities = len(x)
        # print(x)
        dist = 0
        for k in range(n_cities - 1):
            dist += self.dist1[x[k]][x[k + 1]]
        last, first = x[-1], x[0]
        dist += self.dist1[last][first]  # back to the initial city
        return dist

    def get_route_length2(self, x):
        n_cities = len(x)
        dist = 0
        for k in range(n_cities - 1):
            i, j = x[k], x[k + 1]
            dist += self.dist2[i][j]
        last, first = x[-1], x[0]
        dist += self.dist2[last][first]  # back to the initial city
        return dist


class TriTSP(ElementwiseProblem):

    def __init__(self, cities, **kwargs):
        n_cities, _ = cities.shape
        self.cities = cities
        coords1 = cities[:, :2]
        coords2 = cities[:, 2:4]
        coords3 = cities[:, 4:]
        self.dist1 = np.linalg.norm(
                    coords1[None, :, :].repeat(n_cities, axis=0) - coords1[:, None, :].repeat(n_cities, axis=1),
                    axis=-1)
        self.dist2 = np.linalg.norm(
                    coords2[None, :, :].repeat(n_cities, axis=0) - coords2[:, None, :].repeat(n_cities, axis=1),
                    axis=-1)
        self.dist3 = np.linalg.norm(
                    coords3[None, :, :].repeat(n_cities, axis=0) - coords3[:, None, :].repeat(n_cities, axis=1),
                    axis=-1)

        super(TriTSP, self).__init__(
            n_var=n_cities,
            n_obj=3,
            xl=0,
            xu=n_cities - 1,
            type_var=int,
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        L = x
        C = x
        Z = x

        f1 = self.get_route_length1(L)
        f2 = self.get_route_length2(C)
        f3 = self.get_route_length3(Z)

        out['F'] = np.column_stack([f1, f2, f3])

    def get_route_length1(self, x):
        n_cities = len(x)
        # print(x)
        dist = 0
        for k in range(n_cities - 1):
            dist += self.dist1[x[k]][x[k + 1]]
        last, first = x[-1], x[0]
        dist += self.dist1[last][first]  # back to the initial city
        return dist

    def get_route_length2(self, x):
        n_cities = len(x)
        dist = 0
        for k in range(n_cities - 1):
            i, j = x[k], x[k + 1]
            dist += self.dist2[i][j]
        last, first = x[-1], x[0]
        dist += self.dist2[last][first]  # back to the initial city
        return dist

    def get_route_length3(self, x):
        n_cities = len(x)
        dist = 0
        for k in range(n_cities - 1):
            i, j = x[k], x[k + 1]
            dist += self.dist3[i][j]
        last, first = x[-1], x[0]
        dist += self.dist3[last][first]  # back to the initial city
        return dist


class MoVRP(ElementwiseProblem):

    def __init__(self, depot, cities, demand, cap, **kwargs):

        self.depot = depot
        self.cities = cities
        self.coords = np.concatenate((cities, depot.repeat(cities.shape[0] - 1, 0)), axis=0)
        self.demand = np.concatenate((demand, np.array([0])[None, :].repeat(cities.shape[0] - 1, 0)), axis=0)
        self.cap = cap
        self.n_cities = cities.shape[0]
        n_coords = self.coords.shape[0]
        # print(n_coords)
        self.dist = np.linalg.norm(
            self.coords[None, :, :].repeat(n_coords, axis=0) - self.coords[:, None, :].repeat(n_coords, axis=1),
            axis=-1)

        super(MoVRP, self).__init__(
            n_var=n_coords,
            n_obj=2,
            xl=0,
            xu=n_coords - 1,
            type_var=int,
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):

        # L = x
        C = x

        # f1 = self.get_route_length(L)
        f1, f2, penalty = self.get_makespan_and_penalty(C)

        out['F'] = np.column_stack([f1 + penalty, f2 + penalty])

    def get_makespan_and_penalty(self, x):
        # print("original", x is list)
        # print([len(x)-1])
        x = np.concatenate((np.array([len(x) - 1]), x, np.array([len(x) - 1])), axis=0)
        # print("new", x)
        n_cities = len(x)
        makespan = 0
        load_cap = 0
        penalty = 0
        dist = 0
        total_dist = 0
        for k in range(n_cities - 1):
            i, j = x[k], x[k + 1]
            dist += self.dist[i][j]
            total_dist += self.dist[i][j]
            load_cap += self.demand[i]
            if j >= self.n_cities:
                if dist > makespan:
                    makespan = dist
                if load_cap > self.cap:
                    penalty += (load_cap - self.cap) ** 2
                dist = 0
                load_cap = 0
        return total_dist, makespan, penalty

class MoVRPTW(ElementwiseProblem):

    def __init__(self, depot, cities, demand, cap, service_time, tw_start, tw_end, **kwargs):
        self.depot = depot
        self.cities = cities
        self.coords = np.concatenate((cities, depot.repeat(cities.shape[0] - 1, 0)), axis=0)
        self.demand = np.concatenate((demand, np.array([0])[None, :].repeat(cities.shape[0] - 1, 0)), axis=0)
        self.service_time = np.concatenate((service_time, np.array([0])[None, :].repeat(cities.shape[0] - 1, 0)),axis=0)
        self.tw_start = np.concatenate((tw_start, np.array([0])[None, :].repeat(cities.shape[0] - 1, 0)), axis=0)
        self.tw_end = np.concatenate((tw_end, np.array([np.inf])[None, :].repeat(cities.shape[0] - 1, 0)), axis=0)

        self.cap = cap
        self.n_cities = cities.shape[0]
        n_coords = self.coords.shape[0]
        # print(n_coords)
        # print(self.depot.shape)
        # print(self.cities.shape)
        # print(self.coords.shape)
        # print(self.demand.shape)
        # print(self.service_time.shape)
        # print(self.tw_start.shape)
        # print(self.tw_end.shape)
        self.dist = np.linalg.norm(
            self.coords[None, :, :].repeat(n_coords, axis=0) - self.coords[:, None, :].repeat(n_coords, axis=1),
            axis=-1)

        super(MoVRPTW, self).__init__(
            n_var=n_coords,
            n_obj=2,
            xl=0,
            xu=n_coords - 1,
            type_var=int,
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):

        # L = x
        C = x

        # f1 = self.get_route_length(L)
        f1, f2, penalty = self.get_makespan_and_penalty(C)

        out['F'] = np.column_stack([f1 + penalty, f2 + penalty])

    def get_makespan_and_penalty(self, x):
        # print("original", x is list)
        # print([len(x)-1])
        x = np.concatenate((np.array([len(x) - 1]), x, np.array([len(x) - 1])), axis=0)
        # print("new", x)
        n_cities = len(x)
        makespan = 0.0
        load_cap = 0.0
        penalty = 0.0
        dist = 0.0
        total_dist = 0.0
        time = 0.0
        
        for k in range(n_cities - 1):
            i, j = int(x[k]), int(x[k + 1])
            dij = self.dist[i][j]
            dist += dij
            total_dist += dij
            load_cap += self.demand[i, 0]
            time += dij
            
            if j < self.n_cities:
                # 检查到达时间是否超出时间窗口
                ts = float(self.tw_start[j, 0])
                te = float(self.tw_end[j, 0])
                if time < ts:
                    # 计算超出时间窗口的惩罚
                    penalty += (ts - time) ** 3

                elif time > te:
                    # 计算超出时间窗口的惩罚
                    penalty += (time - te) ** 3
                time += self.service_time[j, 0]

            if j >= self.n_cities:
                if dist > makespan:
                    makespan = dist
                if load_cap > self.cap:
                    penalty += (load_cap - self.cap) ** 2
                dist = 0.0
                load_cap = 0.0
                time = 0.0
            
        return total_dist, makespan, penalty

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-objective evolution algorithms for BiTSP, BiVRP and TriTSP")
    parser.add_argument('--problem', default='BiVRPTW', help="The problem to solve (BiTSP, BiVRP, TriTSP)")
    parser.add_argument('--algorithm', default='MOEAD', help="The solving evolution algorithm (MOEAD, NSGA2,NSGA3)")
    parser.add_argument('--graph_size', type=int, default=50, help="The size of the problem graph (20, 50, 100)")
    parser.add_argument('--n_iterations', type=int, default=4000, help='maximum number of generations')

    opts = parser.parse_args()

    save_dir = "{}/{}_benchmark/{}/gens_{}".format(opts.problem, opts.graph_size, opts.algorithm, opts.n_iterations)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if opts.algorithm == "MOEAD":
        ref_dirs = get_reference_directions("uniform", 2, n_partitions=101)

        algorithm = MOEAD(
            ref_dirs,
            n_neighbors=15,
            prob_neighbor_mating=0.7,
            sampling=PermutationRandomSampling(),
            crossover=ERX(),
            mutation=InversionMutation(),
            output=MultiObjectiveOutput()
        )

    elif opts.algorithm == "NSGA2":
        algorithm = NSGA2(
            pop_size=100,
            sampling=PermutationRandomSampling(),
            crossover=ERX(),
            mutation=InversionMutation(),
            eliminate_duplicates=True
        )

    elif opts.algorithm == "NSGA3":
        ref_dirs = get_reference_directions("uniform", 2, n_partitions=101)
        algorithm = NSGA3(
            pop_size=101,
            ref_dirs=ref_dirs,
            sampling=PermutationRandomSampling(),
            crossover=ERX(),
            mutation=InversionMutation(),
            eliminate_duplicates=True
        )

    elif opts.algorithm == "RVEA":
        ref_dirs = get_reference_directions("uniform", 2, n_partitions=101)
        algorithm = RVEA(
            ref_dirs,
            adapt_freq=0,
            pop_size=100,
            sampling=PermutationRandomSampling(),
            crossover=ERX(),
            mutation=InversionMutation(),
            eliminate_duplicates=True
        )

    elif opts.algorithm == 'CTAEA':
        ref_dirs = get_reference_directions("uniform", 2, n_partitions=101)
        algorithm = CTAEA(
            ref_dirs=ref_dirs,
            sampling=PermutationRandomSampling(),
            crossover=ERX(),
            mutation=InversionMutation(),
            eliminate_duplicates=True
        )

    else:
        print('No such algorithm:{}'.format(opts.algorithm))

    termination = get_termination("n_gen", opts.n_iterations)

    if opts.problem == "BiVRP":
        start = time.time()
        loaded_problem = load_dataset('test_data/movrp/movrp%d_test_seed1234.pkl' % (opts.graph_size))[:200]
        for i_index in range(len(loaded_problem)):
            print('Currently this is %d th instance' % (i_index))
            start_time = time.time()
            depot, locs, dem, cap = loaded_problem[i_index]
            depot, locs, dem, cap = np.array(depot), np.array(locs), np.array(dem)[:, None], np.array(cap)
            problem = MoVRP(depot, locs, dem, cap)

            res = minimize(
                problem,
                algorithm,
                termination,
                verbose=False
            )
            np.save(save_dir + '/instance%d.npy' % (i_index), res.F)
            print("The runtime: %d s"%(time.time()-start_time))
        end = time.time()
        print("The total runtime: %d s" % (end - start))

    elif opts.problem == "BiVRPTW":
        start = time.time()
        loaded_problem = load_dataset('./test_data/movrptw/movrptw%d_test_seed1234.pkl' % (opts.graph_size))[:200]
        for i_index in range(len(loaded_problem)):
            print('Currently this is %d th instance' % (i_index))
            start_time = time.time()
            depot, locs, dem, cap, service_time, tw_start, tw_end = loaded_problem[i_index]
            depot, locs, dem, cap, service_time, tw_start, tw_end = np.array(depot), np.array(locs), np.array(dem)[:, None], \
                                                                    np.array(cap), np.array(service_time)[:, None], \
                                                                    np.array(tw_start)[:, None], np.array(tw_end)[:, None]
            problem = MoVRPTW(depot, locs, dem, cap, service_time, tw_start, tw_end)

            res = minimize(
                problem,
                algorithm,
                termination,
                verbose=False
            )
            np.save(save_dir + '/instance%d.npy' % (i_index), res.F)
            print("The runtime: %d s"%(time.time()-start_time))
        end = time.time()
        print("The total runtime: %d s" % (end - start))

    else:
        print("Unknown problem")
