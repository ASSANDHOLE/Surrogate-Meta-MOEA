from typing import List, Any
from copy import deepcopy
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn

from pymoo.core.problem import Problem
from pymoo.problems import get_problem
from pymoo.operators.sampling.lhs import sampling_lhs
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize


CONFIG = {
    'n_var': 10,
    'n_obj': 3,
    'n_sample': 1000,
    'n_epoch': 100,
    'lr': 0.001,
    'lower_bound': 15,
    'upper_bound': 1000,
    'dataset': 'dtlz4',
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
}

DATASET = {}


class TargetModel(nn.Module):
    def __init__(self, n_layer: int, n_arg: int, layer_params: List[int] | np.ndarray):
        super(TargetModel, self).__init__()
        self.n_layer = n_layer
        self.layer_params = layer_params
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_arg, layer_params[0]))
        for i in range(1, n_layer):
            self.layers.append(nn.Linear(layer_params[i - 1], layer_params[i]))
        self.layers.append(nn.Linear(layer_params[-1], 1))

    def forward(self, x):
        for i in range(self.n_layer):
            x = self.layers[i](x)
            x = nn.ReLU()(x)
        x = self.layers[-1](x)
        return x.flatten()


def get_dataset(dataset: str, n_var: int, n_obj: int, n_x: int) -> list:
    # dataset should be pymoo problem
    def tuple_to_tensor(_t: tuple) -> tuple:
        return tuple(torch.tensor(_t[i], dtype=torch.float32, device=CONFIG['device']) for i in range(len(_t)))
    problem = get_problem(dataset, n_var=n_var, n_obj=n_obj)
    x = sampling_lhs(n_samples=n_x, n_var=n_var, xl=0, xu=1)
    f = problem.evaluate(x)
    ret = [(x, f[:, i]) for i in range(n_obj)]
    ret = [tuple_to_tensor(r) for r in ret]
    return ret


def evaluate_model(model: nn.Module, dataset: str) -> List[float]:
    if dataset not in DATASET:
        DATASET[dataset] = get_dataset(dataset, CONFIG['n_var'], CONFIG['n_obj'], CONFIG['n_sample'])
    old_model = model
    loss_fn = nn.MSELoss()
    final_loss = [0.0] * CONFIG['n_obj']
    for i in range(CONFIG['n_obj']):
        x, y = DATASET[dataset][i]
        model = deepcopy(old_model)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
        for e in range(CONFIG['n_epoch']):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            final_loss[i] = loss.item()
            loss.backward()
            optimizer.step()
    return final_loss


class NasProblem(Problem):
    def __init__(self, n_layer: int):
        super().__init__(n_var=n_layer,
                         n_obj=1+CONFIG['n_obj'],
                         n_constr=0,
                         xl=CONFIG['lower_bound'],
                         xu=CONFIG['upper_bound'],
                         vtype=int)
        self.n_layer = n_layer
        self.n_arg = CONFIG['n_var']

    def _evaluate(self, x, out, *args, **kwargs):
        x = x.astype(int)
        fs = []
        for i in range(x.shape[0]):
            model = TargetModel(self.n_layer, self.n_arg, x[i])
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model.to(CONFIG['device'])
            loss_arr = evaluate_model(model, CONFIG['dataset'])
            fs.append([n_params, *loss_arr])
        out['F'] = np.array(fs)


def save_result(data: Any, layer: int, pop_size: int, n_var: int) -> None:
    name = f'results/result-layer_{layer}_pop_{pop_size}_var_{n_var}.pkl'
    with open(name, 'wb') as f:
        try:
            pickle.dump(data, f)
        except Exception as e:
            print('dump failed', e)
            return
    print(f'dump success: {name}')


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', '-l', type=int, default=4)
    parser.add_argument('--pop_size', '-p', type=int, default=50)
    parser.add_argument('--n_gen', '-g', type=int, default=100)
    parser.add_argument('--device', '-d', type=str, default=None)
    args = parser.parse_args()
    if args.device is not None:
        CONFIG['device'] = torch.device(args.device)
    n_layer = args.layer
    pop_size = args.pop_size
    n_gen = args.n_gen
    problem = NasProblem(n_layer)
    n_var = CONFIG['n_var']
    initial_pop = [[2*n_var, *[4*n_var for _ in range(n_layer - 2)], 2*n_var], [100]*n_layer]
    len_remain = 2*pop_size - len(initial_pop)
    remain_pop = np.random.randint(
        low=CONFIG['lower_bound'], high=CONFIG['upper_bound'],
        size=(len_remain, n_layer), dtype=int
    )
    initial_pop = np.concatenate([initial_pop, remain_pop], axis=0)
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=initial_pop,
        crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', n_gen),
        save_history=True,
        verbose=True,
    )
    print(res.X)
    print(res.F)
    save_result(res, n_layer, pop_size, n_var)


if __name__ == '__main__':
    test()
