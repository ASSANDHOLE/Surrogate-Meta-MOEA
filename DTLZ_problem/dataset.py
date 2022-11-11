from __future__ import annotations

from typing import Tuple, List, Any

import numpy as np
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.indicators.igd import IGD
from pymoo.algorithms.moo.nsga3 import NSGA3
from .problem import get_problem


def pf_data(n_var: int, n_objective: int, delta1: int, delta2: int, problem_name: str) -> np.ndarray:
    problem = get_problem(name=problem_name, n_var=n_var, n_obj=n_objective, delta1=delta1, delta2=delta2)  # change delta here
    ref_dirs = get_reference_directions("das-dennis", n_objective, n_partitions=12)
    N = ref_dirs.shape[0]
    # create the algorithm object
    algorithm = NSGA3(pop_size=N, ref_dirs=ref_dirs)
    # execute the optimization
    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', 100))
    return res.X


def create_dataset_inner(x, n_dim: Tuple[int, int], delta: Tuple[List[int], List[int]], problem_name: str) -> Tuple[
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
    problem_name : str
        The problem name

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
        problem = get_problem(name=problem_name, n_var=n_var, n_obj=n_obj, delta1=delta1[i], delta2=delta2[i])
        y.extend([*problem.evaluate(x[i]).transpose()])
    y = np.array(y).astype(np.float32)
    new_x = np.repeat(x, n_obj, axis=0).astype(np.float32)
    return new_x, y


def create_dataset(problem_dim: Tuple[int, int], problem_name: str, x=None, n_problem=None, spt_qry=None, delta=None, 
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
    problem_name : str
        The problem name
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
    def generate_x(_i, _j):
        x_pf = []
        for k in range(n_problem[_i]):
            ps = pf_data(n_var, n_obj, delta[_i][_j][k], delta[_i][_j][k], problem_name)
            x_pf.append(ps[np.random.choice(ps.shape[0], int(0.5 * spt_qry[_j]))])
        x_pf = np.array(x_pf)
        x_ran = np.random.rand(n_problem[_i], spt_qry[_j] - int(0.5 * spt_qry[_j]), n_var)
        return np.concatenate((x_pf, x_ran), axis=1)

    n_var, n_obj = problem_dim

    if delta is not None:
        assert len(delta) == 2
    else:
        # generate delta
        delta = []
        for i in range(2):
            delta1 = np.random.randint(0, 10, n_problem[i])
            delta2 = np.random.randint(0, 10, n_problem[i])
            delta.append([delta1, delta2])

    # generate x
    if x is None:
        x = [None, None, None, None]
    for i in range(2):
        for j in range(2):
            if x[i * 2 + j] is None:
                x[i * 2 + j] = generate_x(i, j)
            # x.append(np.random.rand(n_problem[i], spt_qry[j], n_var))

    train_set = [*create_dataset_inner(x[0], problem_dim, delta[0], problem_name), *create_dataset_inner(x[1], problem_dim, delta[0], problem_name)]
    test_set = [*create_dataset_inner(x[2], problem_dim, delta[1], problem_name), *create_dataset_inner(x[3], problem_dim, delta[1], problem_name)]
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


def evaluate(x: np.ndarray, delta: Tuple[int, int], n_objectives: int, problem_name: str, 
            min_max: Tuple[float | None, float | None]) -> np.ndarray:
    """
    Parameters
    ----------
    x : np.ndarray
        The input data, shape (n_point, n_variables)
    delta : Tuple[int, int]
        The delta1 and delta2
    n_objectives: int
        The number of objectives
    problem_name : str
        The problem name
    min_max : Tuple[float | None, float | None]
        The minimum and maximum of the targets, if None, the targets will not be normalized

    Returns
    -------
    np.ndarray
        The output data, shape (n_point, n_objectives)
    """
    n_variables = x.shape[1]
    problem = get_problem(name=problem_name, n_var=n_variables, n_obj=n_objectives, delta1=delta[0], delta2=delta[1])
    y = problem.evaluate(x)
    if min_max[0] is not None:
        y -= min_max[0]
        y /= min_max[1]
    return y


# def get_pf(n_var: int, n_objectives: int, delta: Tuple[int, int],
#            min_max: Tuple[float | None, float | None]) -> np.ndarray:
#     """
#     Parameters
#     ----------
#     n_var: int
#         The number of variables
#     n_objectives: int
#         The number of objectives
#     delta : Tuple[int, int]
#         The delta1 and delta2
#     min_max : Tuple[float | None, float | None]
#         The minimum and maximum of the targets, if None, the targets will not be normalized

#     Returns
#     -------
#     np.ndarray
#         The parato front, shape (n_point, n_objectives)
#     """
#     problem = DTLZ4c(n_var=n_var, n_obj=n_objectives, delta1=delta[0], delta2=delta[1])
#     ref_dirs = get_reference_directions("das-dennis", n_objectives, n_partitions=12)
#     pf = problem.pareto_front(ref_dirs)
#     if min_max[0] is not None:
#         pf -= min_max[0]
#         pf /= min_max[1]
#     return pf


def get_pf(n_var: int, n_objectives: int, delta: Tuple[int, int], problem_name: str,
           min_max: Tuple[float | None, float | None]) -> np.ndarray:
    """
    Parameters
    ----------
    n_var: int
        The number of variables
    n_objectives: int
        The number of objectives
    delta : Tuple[int, int]
        The delta1 and delta2
    problem_name : str
        The problem name
    min_max : Tuple[float | None, float | None]
        The minimum and maximum of the targets, if None, the targets will not be normalized

    Returns
    -------
    np.ndarray
        The parato front, shape (n_point, n_objectives)
    """
    problem = get_problem(name=problem_name, n_var=n_var, n_obj=n_objectives, delta1=delta[0], delta2=delta[1])  # change delta here
    ref_dirs = get_reference_directions("das-dennis", n_objectives, n_partitions=12)
    N = ref_dirs.shape[0]
    # create the algorithm object
    algorithm = NSGA3(pop_size=N, ref_dirs=ref_dirs)
    # execute the optimization
    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', 100))
    pf = res.F
    if min_max[0] is not None:
        pf -= min_max[0]
        pf /= min_max[1]
    return pf


def get_moea_data(n_var: int, n_objectives: int, delta: Tuple[int, int], algorithm: Any, n_gen: int, metric: Any, problem_name: str,
                  min_max: Tuple[float | None, float | None]) -> Tuple[
    np.ndarray, list, list
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
    metric:
        The metric to calculate the IGD
    problem_name : str
        The problem name
    min_max: Tuple[float | None, float | None]
        The minimum and maximum of the targets, if None, the targets will not be normalized

    Returns
    -------
    Tuple[np.ndarray, list, list]
        moea_pf: The parato front, shape (n_point, n_objectives)
        n_evals: The number of function evaluations
        igd: The IGD
    """
    problem = get_problem(name=problem_name, n_var=n_var, n_obj=n_objectives, delta1=delta[0], delta2=delta[1])  # change delta here
    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', n_gen),
                   save_history=True,
                   verbose=False)
    moea_pf = res.F

    hist = res.history
    hist_F, n_evals = [], []
    for algo in hist:
        n_evals.append(algo.evaluator.n_eval)
        opt = algo.opt
        feas = np.where(opt.get("feasible"))[0]
        hist_F.append(opt.get("F")[feas])
    if min_max[0] is not None:
        for _F in hist_F:
            _F -= min_max[0]
            _F /= min_max[1]
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
