# Unified MAML for this project
from __future__ import annotations

from typing import Tuple, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.indicators.igd import IGD
from pymoo.optimize import minimize

from DTLZ_problem import evaluate, get_pf, get_moea_data
from DTLZ_problem import DTLZbProblem, DTLZ1b
from examples.example import get_args, get_network_structure, get_dataset
from examples.example_sinewave import get_args_maml_regression, get_network_structure_maml_regression, \
    get_dataset_sinewave
from maml_mod import MamlWrapper
from utils import NamedDict
from visualization import visualize_loss, visualize_pf, visualize_igd


def main():
    # see Sol.__init__ for more information
    args = get_args()
    network_structure = get_network_structure(args)
    dataset, _ = get_dataset(args, normalize_targets=True)
    sol = MamlWrapper(dataset, args, network_structure)
    train_loss = sol.train(explicit=False)
    test_loss = sol.test(return_single_loss=False)
    mean_test_loss = np.mean(test_loss, axis=0)
    print(f'Test loss: {mean_test_loss[-1]:.4f}')
    x_test = np.array([i * 0.1 for i in range(1, 1 + 8)], np.float32)
    y_pred = sol(x_test)
    y_true = [y + 1 for y in y_pred]  # add some noise for testing
    sol.test_continue(np.array([x_test] * 3), np.array(y_true, np.float32).reshape((3, 1)))
    y_pred_1 = sol(x_test)
    print(f'Prediction: {y_pred}')
    print(f'Prediction after continue: {y_pred_1}')

    # args.update_step_test = int(1.5 * args.update_step_test)
    sol = MamlWrapper(dataset, args, network_structure)
    random_loss = sol.test(pretrain=True, return_single_loss=False)
    mean_random_loss = np.mean(random_loss, axis=0)
    print(f'Random loss: {mean_random_loss[-1]:.4f}')

    visualize_loss(test_loss, random_loss)


def main_sinewave():
    args = get_args_maml_regression()
    network_structure = get_network_structure_maml_regression()
    dataset = get_dataset_sinewave(args, normalize_targets=True)
    sol = MamlWrapper(dataset, args, network_structure)
    train_loss = sol.train(explicit=5)
    test_loss = sol.test(return_single_loss=False)
    mean_test_loss = np.mean(test_loss, axis=0)
    print(f'Test loss: {mean_test_loss[-1]:.4f}')

    args.update_step_test = int(1.5 * args.update_step_test)
    sol = MamlWrapper(dataset, args, network_structure)
    random_loss = sol.test(return_single_loss=False, pretrain=True)
    print(f'Random loss: {random_loss[-1]:.4f}')
    visualize_loss(mean_test_loss, random_loss)


def main_NSGA():
    args = get_args()
    igd = []
    fn_eval = 0
    fn_eval_limit = 300
    max_pts_num = 5
    pop_size = 60
    sample_size = 30
    n_gen = 10

    network_structure = get_network_structure(args)
    # generate delta
    delta = []
    for i in range(2):
        delta.append([np.random.randint(0, 100, args.train_test[i]), np.random.randint(0, 10, args.train_test[i])])
    dataset, min_max = get_dataset(args, normalize_targets=True, delta=delta)
    sol = MamlWrapper(dataset, args, network_structure)
    train_loss = sol.train(explicit=False)
    test_loss = sol.test(return_single_loss=False)

    n_var = args.problem_dim[0]
    n_objectives = args.problem_dim[1]
    delta_finetune = np.array(delta[1])[:, -1]

    pf_true = get_pf(n_var, n_objectives, delta_finetune, min_max)

    init_x = dataset[1][0][0] # test spt set (100,8)

    res = minimize(DTLZ1b(n_var=n_var, n_obj=n_objectives, delta1=delta_finetune[0], delta2=delta_finetune[1]),
               NSGA2(pop_size=pop_size, sampling=init_x),
               termination=('n_gen', 0.1))

    history_x, history_f = res.X, res.F
    history_f -= min_max[0]
    history_f /= min_max[1]

    metric = IGD(pf_true, zero_to_one=True)
    igd.append(metric.do(history_f))

    while fn_eval < fn_eval_limit:

        # if history_x.shape[0] < sample_size:
        #     random_x = np.random.rand(pop_size - history_x.shape[0], n_var)
        #     sample_x = np.vstack((history_x, random_x))
        # else:
        #     random_x = np.random.rand(pop_size - sample_size, n_var)
        #     sample_x = np.vstack((history_x[np.random.choice(history_x.shape[0], sample_size)], random_x))

        algorithm = NSGA2(pop_size=pop_size, sampling=history_x)

        res = minimize(DTLZbProblem(sol=sol),
                       algorithm,
                       ("n_gen", n_gen),
                       seed=1,
                       verbose=False)
        
        X = res.X
        if len(X) > max_pts_num:
            X = X[np.random.choice(X.shape[0], max_pts_num)]
        
        history_x = np.vstack((history_x, X))
        # history_f = np.vstack((history_f, res.F))

        fn_eval += X.shape[0]

        X = X.astype(np.float32)
        y_true = evaluate(X, delta_finetune, n_objectives, min_max=min_max)

        history_f = np.vstack((history_f, y_true))

        new_y_true = []
        for i in range(n_objectives):
            new_y_true.append(y_true[:, i])
        new_y_true = np.array(new_y_true, dtype=np.float32)
        new_y_true = new_y_true.reshape((*new_y_true.shape, 1))

        sol.test_continue(X, new_y_true)

        metric = IGD(pf_true, zero_to_one=True)
        igd.append(metric.do(history_f))
    
    pf = evaluate(res.X, delta_finetune, n_objectives, min_max=min_max)
    moea_pf, n_evals_moea, igd_moea = get_moea_data(n_var, n_objectives, delta_finetune, NSGA2(pop_size=pop_size, sampling=init_x), int(fn_eval_limit/pop_size), pf_true, min_max)
    n_evals_moea = np.insert(n_evals_moea, 0, 0)
    igd_moea = np.insert(igd_moea, 0, igd[0])

    visualize_pf(pf=pf, label='Sorrogate PF', color='green', scale=[0.5]*3, pf_true=pf_true,
                 show=SHOW_PLOT, save_path=f'{PREFIX}sor_pf.png', save_alternative=True)
    visualize_pf(pf=moea_pf, label='NSGA-II PF', color='blue', scale=[0.5]*3, pf_true=pf_true,
                 show=SHOW_PLOT, save_path=f'{PREFIX}moea_pf.png', save_alternative=True)

    func_evals = [max_pts_num*np.arange(len(igd)), n_evals_moea]
    igds = [igd, igd_moea]
    colors = ['black', 'blue']
    labels = ["Our Surrogate Model", "NSGA-II"]
    visualize_igd(func_evals, igds, colors, labels,
                  show=SHOW_PLOT, save_path=f'{PREFIX}igd.png', save_alternative=True)
    plt.show()


if __name__ == '__main__':
    # main()
    # main_sinewave()
    PREFIX = 'imgs/'
    SHOW_PLOT = False
    main_NSGA()

