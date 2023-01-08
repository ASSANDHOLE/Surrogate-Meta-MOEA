import os
import pickle
import sys
import time
from multiprocessing import Pool
from multiprocessing import Manager as LockManager
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('..')

import random
import torch

from benchmarking import GpuManager

import numpy as np
from pymoo.indicators.igd import IGD
from DTLZ_problem import DTLZbProblem, get_custom_problem
from DTLZ_problem import evaluate, get_pf
from benchmarking import benchmark_for_seeds_different_args
from problem_config.example import get_args, get_dataset
from maml_mod import MamlWrapperMrA as MamlWrapper
from pymoo.core.problem import Problem
from pymoo.operators.sampling.lhs import sampling_lhs
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize

CONFIG = {
    'layer': 3,
    'seeds': None,
    'n_gpu': 1,
    'n_parallel': int(1),
    'gpu_sel_param': 0
}


if os.path.exists('./config.json'):
    import json
    with open('./config.json', 'r') as _f:
        CONFIG.update(json.load(_f))


CACHE = {}


def net_structure(args, structure_list: List[int]):
    n_args_in = args.problem_dim[0]
    conf = [('linear', [structure_list[0], n_args_in]), ('relu', [True])]
    for i in range(1, len(structure_list)):
        conf.append(('linear', [structure_list[i], structure_list[i - 1]]))
        conf.append(('relu', [True]))
    conf.append(('linear', [1, structure_list[-1]]))
    return conf


def has_local_storage(name: str):
    return os.path.exists(name)


def load_local_storage(name: str):
    try:
        with open(name, 'rb') as f:
            return pickle.load(f)
    except pickle.PickleError as e:
        if os.path.exists(name):
            try:
                os.remove(name)
            except OSError:
                pass
        return None


def save_to_local_storage(name: str, data):
    with open(name, 'wb') as f:
        pickle.dump(data, f)


def cache_dataset(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed in CACHE:
        return CACHE[seed]
    name = f'datastore/dataset_{seed}.pkl'
    if has_local_storage(name):
        val = load_local_storage(name)
        if val is not None:
            dataset, min_max, delta, x = val
            CACHE[seed] = dataset, min_max, delta, x
            return dataset, min_max, delta, x
    args = get_args()
    n_var = args.problem_dim[0]
    delta = []
    for i in range(2):
        delta.append([np.random.randint(0, 100, args.train_test[i]), np.random.randint(0, 10, args.train_test[i])])
    x = [None, None, None, None]
    x[2] = sampling_lhs(n_samples=11 * n_var - 1, n_var=n_var, xl=0, xu=1)
    x[2] = x[2][np.random.choice(x[2].shape[0], args.k_spt, replace=False), :]
    x[2] = x[2].reshape((1, *x[2].shape))
    dataset, min_max = get_dataset(args, normalize_targets=True, x=x, delta=delta, problem_name='DTLZ4c')
    CACHE[seed] = dataset, min_max, delta, x
    save_to_local_storage(name, (dataset, min_max, delta, x))
    return dataset, min_max, delta, x


def worker(gpu_id: int, seed: int, net_structure_list: List[int]):
    args = get_args()
    args.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    n_var = args.problem_dim[0]
    n_objectives = args.problem_dim[1]
    igd = []
    fn_eval = args.k_spt
    fn_eval_limit = 200 - 2
    max_pts_num = 10
    proxy_n_gen = 50
    proxy_pop_size = 50
    problem_name = 'DTLZ4c'

    network_structure = net_structure(args, net_structure_list)
    # generate delta
    dataset, min_max, delta, x_init = cache_dataset(seed)
    sol = MamlWrapper(dataset, args, network_structure)
    sol.train(explicit=False)
    sol.test(return_single_loss=False)

    delta_finetune = np.array(delta[1])[:, -1]

    init_x = dataset[1][0][0]  # test spt set (100, 8)

    problem = get_custom_problem(name=problem_name, n_var=n_var, n_obj=n_objectives, delta1=delta_finetune[0],
                          delta2=delta_finetune[1])

    pf_true = get_pf(n_objectives, problem, min_max)

    res = minimize(problem=problem,
                   algorithm=NSGA2(pop_size=proxy_pop_size, sampling=init_x),
                   termination=('n_gen', 0.1))

    history_x, history_f = res.X, res.F
    history_x = history_x.astype(np.float32)
    history_f = history_f.astype(np.float32)
    history_f -= min_max[0]
    history_f /= min_max[1]

    metric = IGD(pf_true, zero_to_one=True)
    igd.append(metric.do(history_f))
    igd.append(igd[-1])

    while fn_eval < fn_eval_limit:
        algorithm_surrogate = NSGA2(pop_size=args.k_spt, sampling=history_x)
        problem_surrogate = DTLZbProblem(n_var=n_var, n_obj=n_objectives, sol=sol)

        res = minimize(problem_surrogate,
                       algorithm_surrogate,
                       ('n_gen', proxy_n_gen),
                       verbose=False)

        x = res.X

        if len(x) > max_pts_num:
            x = x[np.random.choice(x.shape[0], max_pts_num)]
        x = x.astype(np.float32)

        history_x = np.vstack((history_x, x))

        fn_eval += x.shape[0]

        y_true = evaluate(x, delta_finetune, n_objectives, problem_name, min_max=min_max)
        y_true = y_true.astype(np.float32)

        history_f = np.vstack((history_f, y_true))

        reshaped_history_f = []
        for i in range(n_objectives):
            reshaped_history_f.append(history_f[:, i])
        reshaped_history_f = np.array(reshaped_history_f, dtype=np.float32)
        reshaped_history_f = reshaped_history_f.reshape((*reshaped_history_f.shape, 1))

        sol.test_continue(history_x, reshaped_history_f, return_single_loss=True)

        igd.append(metric.do(history_f))

    del sol
    return igd[-1]


def post_mean_max(data: list | np.ndarray):
    return np.mean(data), np.max(data)


def runner(data):
    gpu_sel, lock, seed, net_structure_list = data
    gpu_id = -1
    if isinstance(gpu_sel, int):
        gpu_id = gpu_sel
    else:
        while gpu_id < 0:
            try:
                gpu_id = gpu_sel.get_gpu(lock=lock)
            except RuntimeError:
                time.sleep(1)
    print(f'gpu_id: {gpu_id}', flush=True)
    res = worker(gpu_id, seed, net_structure_list)
    if not isinstance(gpu_sel, int):
        gpu_sel.return_gpu(gpu_id, lock=lock)
    return res


def bench_runner(data):
    func, post_func, args = data
    args = [[(*args[0:2], s, args[3])] for s in args[2]]
    ret = benchmark_for_seeds_different_args(func, post_func, seeds=CONFIG['seeds'], func_args=args)
    return ret


class NasProblem(Problem):
    def __init__(self):
        super().__init__(n_var=CONFIG['layer'],
                         n_obj=2,
                         n_constr=0,
                         xl=15,
                         xu=500,
                         vtype=int)

    def _evaluate(self, x, out, *args, **kwargs):
        seeds = CONFIG['seeds']
        with GpuManager() as gpu_manager:
            with LockManager() as lock_manager:
                the_lock = lock_manager.Lock()
                n_gpu = CONFIG['n_gpu']
                gpu_sel_param = CONFIG['gpu_sel_param']
                if n_gpu > 1:
                    gpu_sel_param = gpu_sel_param if gpu_sel_param is not None else tuple()
                    gpu_selector = gpu_manager.GpuSelector(n_gpu, *gpu_sel_param)
                else:
                    gpu_selector = gpu_sel_param if gpu_sel_param is not None else 0
                params = [(gpu_selector, the_lock, seeds, x[i]) for i in range(x.shape[0])]
                params = [(runner, post_mean_max, param) for param in params]
                n_para = CONFIG['n_parallel']
                if n_para == 1:
                    # no multiprocessing
                    res = [bench_runner(param) for param in params]
                else:
                    with Pool(CONFIG['n_parallel'], maxtasksperchild=1) as p:
                        res = p.map(bench_runner, params)
        res = np.array(res)
        out['F'] = res


def main():
    pop_size = 50
    n_gen = 100
    the_seed = 42

    random.seed(the_seed)
    seeds = random.sample(range(100000), 50)
    CONFIG['seeds'] = seeds
    for seed in seeds:
        cache_dataset(seed)
    random.seed(the_seed)
    np.random.seed(the_seed)
    problem = NasProblem()
    initial_pop = [[15, 15, 15]]
    len_remain = 2 * pop_size - len(initial_pop)
    remain_pop = np.random.randint(
        low=10, high=500,
        size=(len_remain, CONFIG['layer']), dtype=int
    )

    initial_pop = np.concatenate([initial_pop, remain_pop], axis=0)
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=initial_pop,
        crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )
    print('Init OK')
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', n_gen),
        save_history=True,
        verbose=True,
    )
    print(res.X)
    print(res.F)
    try:
        os.makedirs('results')
    except FileExistsError:
        pass
    try:
        with open('results/nas_results.pkl', 'wb') as f:
            pickle.dump(res, f)
    except Exception as e:
        print(f'Pickle Dump Error: {e}')


def single_model_run():
    n_args = (10, 3)

    class Net(nn.Module):
        def __init__(self, s):
            super(Net, self).__init__()
            net = [
                nn.Linear(n_args[0], s[0]),
            ]
            for i in range(len(s) - 1):
                net.append(nn.ReLU())
                net.append(nn.Linear(s[i], s[i + 1]))
            net.append(nn.ReLU())
            net.append(nn.Linear(s[-1], n_args[1]))
            self.net = nn.Sequential(*net)

        def forward(self, x):
            return self.net(x)

    net = Net([100, 200, 200, 200, 100])
    args = get_args()
    args.k_spt = 2000
    args.k_qry = 2000
    dataset, norm = get_dataset(args,
                                normalize_targets=True,
                                problem_name='DTLZ4c',
                                pf_ratio=0,
                                dim=1)
    train_x, train_y, test_x, test_y = dataset[1]
    dev = torch.device('cuda:0')

    def remove_one_from_shape(x):
        s = list(x.shape)
        s = [ss for ss in s if ss != 1]
        x = x.reshape(s)
        return torch.from_numpy(x).float().to(dev)

    train_x = remove_one_from_shape(train_x)
    train_y = remove_one_from_shape(train_y)
    test_x = remove_one_from_shape(test_x)
    test_y = remove_one_from_shape(test_y)

    net = net.to(dev)

    # train
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for i in range(207):
        optimizer.zero_grad()
        random_idx = np.random.choice(train_x.shape[0], 100)
        x = train_x[random_idx]
        y = train_y[random_idx]
        y_pred = net(x)
        loss = F.mse_loss(y_pred, y)
        loss.backward()
        optimizer.step()
        print(f'Epoch {i}: {loss.item()}')

    # test
    net.eval()
    y_pred = net(test_x)
    loss = F.mse_loss(y_pred, test_y)
    print(f'Test Loss: {loss.item()}')


if __name__ == '__main__':
    single_model_run()
