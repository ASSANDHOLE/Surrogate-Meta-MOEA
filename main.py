# Unified MAML for this project
from __future__ import annotations

import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.igd import IGD
from pymoo.operators.sampling.lhs import sampling_lhs
from pymoo.optimize import minimize

from DTLZ_problem import DTLZbProblem, get_custom_problem
from DTLZ_problem import evaluate, get_pf, get_moea_data
from benchmarking import benchmark_for_seeds
from problem_config.example import get_args, get_network_structure, get_dataset, estimate_resource_usage
from maml_mod import MamlWrapper
from utils import NamedDict, set_ipython_exception_hook
from visualization import visualize_loss, visualize_pf, visualize_igd


def cprint(*args, do_print=True, **kwargs):
    if do_print:
        print(*args, **kwargs)


def test():
    # see Sol.__init__ for more information
    args = get_args()
    network_structure = get_network_structure(args)
    dataset, _ = get_dataset(args, normalize_targets=True, problem_name='DTLZ4c')
    sol = MamlWrapper(dataset, args, network_structure)
    # train_loss = sol.train(explicit=1)
    test_loss = sol.test(return_single_loss=False)
    mean_test_loss = np.mean(test_loss, axis=0)
    print(f'Test loss: {mean_test_loss[-1]:.4f}')
    x_test = dataset[1][2][1]
    y_true = dataset[1][3][1]
    y_pred = [sol(x)[1] for x in x_test]
    print(y_true[:10])
    print(y_pred[:10])
    x_test = np.array([i * 0.09 for i in range(1, 1 + 10)], np.float32)
    y_pred = sol(x_test)
    y_true = [y + 1 for y in y_pred]  # add some noise for testing
    sol.test_continue(x_test, np.array(y_true, np.float32).reshape((3, 1)))
    y_pred_1 = sol(x_test)
    print(f'Prediction: {y_pred}')
    print(f'Prediction after continue: {y_pred_1}')

    # args.update_step_test = int(1.5 * args.update_step_test)
    sol = MamlWrapper(dataset, args, network_structure)
    random_loss = sol.test(pretrain=True, return_single_loss=False)
    mean_random_loss = np.mean(random_loss, axis=0)
    print(f'Random loss: {mean_random_loss[-1]:.4f}')

    visualize_loss(test_loss, random_loss)


def main(problem_name: str, print_progress=False, do_plot=False, do_train=True, gpu_id: int | None = None):
    args = get_args()
    if gpu_id is not None:
        args.device = torch.device(f'cuda:{gpu_id}')
    n_var = args.problem_dim[0]
    n_objectives = args.problem_dim[1]
    igd = []
    fn_eval = args.k_spt
    fn_eval_limit = 200 - 2
    max_pts_num = 10
    moea_pop_size = 30
    proxy_n_gen = 50
    proxy_pop_size = 50

    network_structure = get_network_structure(args)
    # generate delta
    delta = []
    for i in range(2):
        delta.append([np.random.rand(args.train_test[i])*8, np.random.rand(args.train_test[i])*8])
    x = [None, None, None, None]
    x[2] = sampling_lhs(n_samples=11 * n_var - 1, n_var=n_var, xl=0, xu=1)
    # sample 'arg.k_spt' from x[2]
    x[2] = x[2][np.random.choice(x[2].shape[0], args.k_spt, replace=False), :]
    dataset, min_max = get_dataset(args, normalize_targets=True, delta=delta, problem_name=problem_name, pf_ratio=0)
    sol = MamlWrapper(dataset, args, network_structure)
    cprint('dataset init complete', do_print=print_progress)
    if do_train:
        train_loss = sol.train(explicit=print_progress)
        print(train_loss[-1])
    test_loss = sol.test(return_single_loss=False)
    cprint('MAML init complete', do_print=print_progress)

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

    # only for visualization
    Y_igd = []
    Y_igd.append(metric.do(history_f))
    Y_igd.append(igd[-1])

    func_eval_igd = [0, fn_eval]

    cprint('Algorithm init complete', do_print=print_progress)

    plot_int = 30
    plotted = 1

    while fn_eval < fn_eval_limit:
        cprint(f'fn_eval: {fn_eval}', do_print=print_progress)
        algorithm_surrogate = NSGA2(pop_size=args.k_spt, sampling=history_x)
        problem_surrogate = DTLZbProblem(n_var=n_var, n_obj=n_objectives, sol=sol)

        res = minimize(problem_surrogate,
                       algorithm_surrogate,
                       ('n_gen', proxy_n_gen),
                       verbose=False)

        X = res.X
        # only for visualization
        Y_true = evaluate(X, delta_finetune, n_objectives, problem_name, min_max=min_max)
        Y_igd.append(metric.do(Y_true))

        if len(X) > max_pts_num:
            X = X[np.random.choice(X.shape[0], max_pts_num)]
        X = X.astype(np.float32)

        history_x = np.vstack((history_x, X))
        # history_f = np.vstack((history_f, res.F))

        fn_eval += X.shape[0]

        y_true = evaluate(X, delta_finetune, n_objectives, problem_name, min_max=min_max)
        y_true = y_true.astype(np.float32)

        history_f = np.vstack((history_f, y_true))

        cont_loss = sol.test_continue(history_x, history_f.T, return_single_loss=True)
        # cont_loss = sol.test_continue(X, y_true.T, return_single_loss=True)
        cprint(f'continue loss: {cont_loss}', do_print=print_progress)

        # metric = IGD(pf_true, zero_to_one=True)
        igd.append(metric.do(history_f))
        func_eval_igd.append(fn_eval)

        pf_true_surrogate = get_pf(n_objectives, problem_surrogate)
        scale = []
        for i in range(n_objectives):
            concatenated = np.concatenate([pf_true_surrogate[:, i], pf_true[:, i]]),
            data = [np.min(concatenated), np.max(concatenated)]
            scale.append(data)
        if fn_eval > plotted + plot_int:
            plotted = fn_eval
            visualize_pf(pf=pf_true_surrogate, label='Surrogate PF', color='green',
                         scale=scale, pf_true=pf_true, show=True)

    # pf = evaluate(res.X, delta_finetune, n_objectives, min_max=min_max)
    cprint('Algorithm complete', do_print=print_progress)
    pf = history_f
    moea_problem = NSGA2(pop_size=moea_pop_size, sampling=init_x)
    moea_pf, n_evals_moea, igd_moea = get_moea_data(n_var, n_objectives, delta_finetune,
                                                    moea_problem,
                                                    int(fn_eval_limit / moea_pop_size), metric, problem_name, min_max)
    n_evals_moea = np.insert(n_evals_moea, 0, 0)
    igd_moea = np.insert(igd_moea, 0, igd[0])
    cprint('MOEA Baseline complete', do_print=print_progress)

    if do_plot:
        scale = []
        for i in range(n_objectives):
            concatenated = np.concatenate([pf[:, i], pf_true[:, i]]),
            data = [np.min(concatenated), np.max(concatenated)]
            scale.append(data)
        visualize_pf(pf=pf, label='Surrogate PF', color='green', scale=scale, pf_true=pf_true)
        scale = []
        for i in range(n_objectives):
            concatenated = np.concatenate([moea_pf[:, i], pf_true[:, i]]),
            data = [np.min(concatenated), np.max(concatenated)]
            scale.append(data)
        visualize_pf(pf=moea_pf, label='NSGA-II PF', color='blue', scale=scale, pf_true=pf_true)

    func_evals = [func_eval_igd, n_evals_moea, func_eval_igd]
    igds = [igd, igd_moea, Y_igd]
    colors = ['black', 'blue', 'green']
    labels = ['Our Surrogate Model', 'NSGA-II', 'Test']
    if do_plot:
        visualize_igd(func_evals, igds, colors, labels)
        plt.show()

    cprint(f'IGD of Proxy: {igd[-3:-1]}', do_print=print_progress)
    cprint(f'IGD of MOEA:  {igd_moea[-3:-1]}', do_print=print_progress)
    # deallocate memory
    del sol
    return igd[-1]


def post_mean_std(data: list | np.ndarray):
    return np.mean(data), np.std(data)


def usage_check(n_proc: int):
    def parse_value(_val):
        # unit: GB
        unit = _val[-1]
        if unit == 'G':
            return float(_val[:-1])
        elif unit == 'M':
            return float(_val[:-1]) / 1024
        elif unit == 'K':
            return float(_val[:-1]) / 1024 / 1024
        else:
            # assume it is in B
            return float(_val) / 1024 / 1024 / 1024

    if n_proc < 0:
        import os
        n_proc = os.cpu_count() + n_proc
        if n_proc is None:
            n_proc = 1
    est = estimate_resource_usage()
    mem_per_proc = parse_value(est.memory)
    gpu_mem_per_proc = parse_value(est.gpu_memory)
    desc = est.description
    total_meme_usage = mem_per_proc * n_proc
    total_gpu_meme_usage = gpu_mem_per_proc * n_proc
    print(f'Estimated RAM usage: {total_meme_usage:.2f} GB')
    print(f'Estimated GPU RAM usage: {total_gpu_meme_usage:.2f} GB')
    print(desc)
    print('if the estimated usage is too large, you may cause system crash, try to reduce the number of processes')


def main_benchmark(problem_name: str):
    _seeds = 20
    _n_proc = 20
    init_seed = 42
    _estimate_gram = 3.5
    usage_check(_n_proc)
    gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    _res = benchmark_for_seeds(main,
                               post_mean_std,
                               seeds=_seeds,
                               func_args=[problem_name],
                               func_kwargs={'print_progress': False, 'do_train': True},
                               n_proc=_n_proc,
                               gpu_ids=gpu_ids,
                               estimated_gram=_estimate_gram,
                               init_seed=init_seed)
    print(f'MAML Trained IGD: {_res[0]} +- {_res[1]}')
    _res = benchmark_for_seeds(main,
                               post_mean_std,
                               seeds=_seeds,
                               func_args=[problem_name],
                               func_kwargs={'print_progress': False, 'do_train': False},
                               n_proc=_n_proc,
                               gpu_ids=gpu_ids,
                               estimated_gram=_estimate_gram,
                               init_seed=init_seed)
    print(f'NON-Trained  IGD: {_res[0]} +- {_res[1]}')


def fast_seed(seed: int) -> None:
    import numpy as np
    import random
    import torch
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # set_ipython_exception_hook()
    # fast_seed(20010808)
    # main()
    # main_NSGA_4c(do_plot=True, print_progress=True, do_train=False)
    problems = NamedDict({
        'd1': 'DTLZ1b',
        'd4': 'DTLZ4c',
        'd7': 'DTLZ7b',
    })
    # main(problems.d7, do_plot=True, print_progress=True, do_train=True)
    main_benchmark(problems.d1)
