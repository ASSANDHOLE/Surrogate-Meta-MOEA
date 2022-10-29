from __future__ import annotations

from typing import Tuple, List

import numpy as np
from pymoo.core.problem import Problem
from pymoo.problems.many.dtlz import get_ref_dirs
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.indicators.igd import IGD
from pymoo.algorithms.moo.nsga3 import NSGA3


class DTLZb(Problem):
    def __init__(self, n_var, n_obj, delta1, delta2, k=None):
        self.delta1 = delta1
        self.delta2 = delta2

        if n_var:
            self.k = n_var - n_obj + 1
        elif k:
            self.k = k
            n_var = k + n_obj - 1
        else:
            raise Exception("Either provide number of variables or k!")

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=0, xu=1, type_var=np.double)

    def g1(self, X_M):
        return (100 + self.delta1) * (
                self.k + self.delta2 + np.sum(np.square(X_M - 0.5) - np.cos(20 * np.pi * (X_M - 0.5)), axis=1))

    def g2(self, X_M):
        return np.sum(np.square(X_M - 0.5), axis=1)

    def obj_func(self, X_, g, alpha=1):
        f = []

        for i in range(0, self.n_obj):
            _f = (1 + g)
            _f *= np.prod(np.cos(np.power(X_[:, :X_.shape[1] - i], alpha) * np.pi / 2.0), axis=1)
            if i > 0:
                _f *= np.sin(np.power(X_[:, X_.shape[1] - i], alpha) * np.pi / 2.0)

            f.append(_f)

        f = np.column_stack(f)
        return f


class DTLZ1b(DTLZb):
    def __init__(self, n_var=7, n_obj=3, delta1=0, delta2=0, **kwargs):
        self.delta1 = delta1
        self.delta2 = delta2
        super().__init__(n_var, n_obj, delta1, delta2, **kwargs)

    def _calc_pareto_front(self, ref_dirs=None):
        if ref_dirs is None:
            ref_dirs = get_ref_dirs(self.n_obj)
        coefficient = max(0.5, 0.5*(100+self.delta1)*self.delta2)
        return coefficient * ref_dirs

    def obj_func(self, X_, g):
        f = []

        for i in range(0, self.n_obj):
            _f = 0.5 * (1 + g)
            _f *= np.prod(X_[:, :X_.shape[1] - i], axis=1)
            if i > 0:
                _f *= 1 - X_[:, X_.shape[1] - i]
            f.append(_f)

        return np.column_stack(f)

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)
        out["F"] = self.obj_func(X_, g)


def pf_data(n_var: int, n_objective: int, delta1: int, delta2: int) -> np.ndarray:
    problem = DTLZ1b(n_var=n_var, n_obj = n_objective, delta1=delta1, delta2=delta2) # change delta here
    ref_dirs = get_reference_directions("das-dennis", n_objective, n_partitions=12)
    N = ref_dirs.shape[0]
    # create the algorithm object
    algorithm = NSGA3(pop_size=N, ref_dirs=ref_dirs)
    # execute the optimization
    res = minimize(problem,
                algorithm,
                termination=('n_gen', 100))
    return res.X


def create_dataset_inner(x, n_dim: Tuple[int, int], delta: Tuple[List[int], List[int]]) -> Tuple[
    np.ndarray, np.ndarray
]:
    """
    Parameters
    ----------
    x : np.ndarray
        The input data, shape (n_problem, n_spt, n_variables)
    n_dim : Tuple[int, int]
        The number of variables and number of objectives
    delta : Tuple[List[int], List[int]]
        The delta1 and delta2 for each problem

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        [x, y]
    """
    delta1, delta2 = delta
    n_problem = len(delta1)
    n_var, n_obj = n_dim
    y = []
    for i in range(n_problem):
        problem = DTLZ1b(n_var=n_var, n_obj=n_obj, delta1=delta1[i], delta2=delta2[i])
        y.extend([*problem.evaluate(x[i]).transpose()])
    y = np.array(y).astype(np.float32)
    new_x = np.repeat(x, n_obj, axis=0).astype(np.float32)
    return new_x, y


def create_dataset(problem_dim: Tuple[int, int], x=None, n_problem=None, spt_qry=None, delta=None,
                   normalize_targets=True, **_) -> Tuple[
    Tuple[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ], Tuple[float | None, float | None]]:
    """
    Parameters
    ----------
    problem_dim : Tuple[int, int]
        The number of variables and number of objectives for each problem
    x : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        [x_spt_train, x_qry_train, x_spt_test, x_qry_test]
        The input data, shape (4, n_problem, n_spt, n_variables)
    delta : Tuple[train_delta,  test_delta]
        The delta values, shape (2, 2, n_problem)
    n_problem : Tuple[int, int]
        [n_train, n_test]
    spt_qry : Tuple[int, int]
        The number of support and query points for each problem
    normalize_targets : bool
        Whether to normalize the targets

    Returns
    -------
    Tuple[
        Tuple[
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ],
        Tuple[float | None, float | None]
    ]
        In the first Tuple:
            The first element is the training set [support set, support label, query set, query label]
            The second element is the test set [support set, support label, query set, query label]
        In the second Tuple:
            The minimum and maximum of the targets (to be used for normalization)
    """
    n_var, n_obj = problem_dim

    if delta is not None:
        assert len(delta) == 2
    else:
        # generate delta
        delta = []
        for i in range(2):
            delta1 = np.random.randint(0, 100, n_problem[i])
            delta2 = np.random.randint(0, 10, n_problem[i])
            delta.append([delta1, delta2])
    
    if x is not None:
        assert len(x) == 4
    else:
        # generate x
        x = []
        for i in range(2):
            for j in range(2):
                x_pf = []
                for k in range(n_problem[i]):
                    ps = pf_data(n_var, n_obj, delta[i][j][k], delta[i][j][k])
                    x_pf.append(ps[np.random.choice(ps.shape[0], int(0.5 * spt_qry[j]))])
                x_pf = np.array(x_pf)
                x_ran = np.random.rand(n_problem[i], spt_qry[j] - int(0.5 * spt_qry[j]), n_var)
                x.append(np.concatenate((x_pf, x_ran), axis=1))
                # x.append(np.random.rand(n_problem[i], spt_qry[j], n_var))

    train_set = [*create_dataset_inner(x[0], problem_dim, delta[0]), *create_dataset_inner(x[1], problem_dim, delta[0])]
    test_set = [*create_dataset_inner(x[2], problem_dim, delta[1]), *create_dataset_inner(x[3], problem_dim, delta[1])]
    if normalize_targets:
        minimum = np.min(np.concatenate([train_set[1], test_set[1]]), axis=None)
        train_set[1] -= minimum
        test_set[1] -= minimum
        train_set[3] -= minimum
        test_set[3] -= minimum
        maximum = np.max(np.concatenate([train_set[1], test_set[1]]), axis=None)
        train_set[1] /= maximum
        test_set[1] /= maximum
        train_set[3] /= maximum
        test_set[3] /= maximum
    else:
        maximum = None
        minimum = None
    return (tuple(train_set), tuple(test_set)), (minimum, maximum)


def evaluate(x: np.ndarray, delta: Tuple[int, int],
             n_objectives: int, min_max: Tuple[float | None, float | None]) -> np.ndarray:
    """
    Parameters
    ----------
    x : np.ndarray
        The input data, shape (n_point, n_variables)
    delta : Tuple[int, int]
        The delta1 and delta2
    n_objectives: int
        The number of objectives
    min_max : Tuple[float | None, float | None]
        The minimum and maximum of the targets, if None, the targets will not be normalized

    Returns
    -------
    np.ndarray
        The output data, shape (n_point, n_objectives)
    """
    n_variables = x.shape[1]
    problem = DTLZ1b(n_var=n_variables, n_obj=n_objectives, delta1=delta[0], delta2=delta[1])
    y = problem.evaluate(x)
    if min_max[0] is not None:
        y -= min_max[0]
        y /= min_max[1]
    return y

def get_pf(n_var: int, n_objectives: int, delta: Tuple[int, int], min_max: Tuple[float | None, float | None]) -> np.ndarray:
    """
    Parameters
    ----------
    n_var: int
        The number of variables
    n_objectives: int
        The number of objectives
    delta : Tuple[int, int]
        The delta1 and delta2
    min_max : Tuple[float | None, float | None]
        The minimum and maximum of the targets, if None, the targets will not be normalized

    Returns
    -------
    np.ndarray
        The parato front, shape (n_point, n_objectives)
    """
    problem = DTLZ1b(n_var = n_var, n_obj = n_objectives, delta1=delta[0], delta2=delta[1])
    ref_dirs = get_reference_directions("das-dennis", n_objectives, n_partitions=12)
    pf = problem.pareto_front(ref_dirs)
    if min_max[0] is not None:
        pf -= min_max[0]
        pf /= min_max[1]
    return pf

def get_moea_data(n_var: int, n_objectives: int, delta: Tuple[int, int], algorithm, n_gen: int, min_max: Tuple[float | None, float | None]) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Parameters
    ----------
    n_var: int
        The number of variables
    n_objectives: int
        The number of objectives
    delta: Tuple[int, int]
        The delta1 and delta2
    algorithm: 
        MOEA algorithm
    n_gen: int
        number of generation
    min_max: Tuple[float | None, float | None]
        The minimum and maximum of the targets, if None, the targets will not be normalized

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        moea_pf: The parato front, shape (n_point, n_objectives)
        n_evals: The number of function evaluations
        igd: The IGD
    """
    problem = DTLZ1b(n_var = n_var, n_obj = n_objectives, delta1=delta[0], delta2=delta[1]) # change delta here
    res = minimize(problem,
                algorithm,
                termination=('n_gen', n_gen),
                save_history=True,
                verbose=False)
    moea_pf = res.F

    hist = res.history
    hist_F,n_evals = [],[]
    for algo in hist:
        n_evals.append(algo.evaluator.n_eval)
        opt = algo.opt
        feas = np.where(opt.get("feasible"))[0]
        hist_F.append(opt.get("F")[feas])
    metric = IGD(moea_pf, zero_to_one=True)
    igd = [metric.do(_F) for _F in hist_F]

    if min_max[0] is not None:
        moea_pf -= min_max[0]
        moea_pf /= min_max[1]
    return moea_pf, n_evals, igd

def test():
    n_dim = (8, 3)
    train_set, test_set = create_dataset(n_dim, n_problem=(4, 2), spt_qry=(5, 20))  # (12, 5, 7)
    print(train_set[0].shape, train_set[1].shape, test_set[0].shape, test_set[1].shape)


if __name__ == '__main__':
    test()
