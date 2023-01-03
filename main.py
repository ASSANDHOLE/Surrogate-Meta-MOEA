from __future__ import annotations

from typing import List, Literal

import numpy as np
import torch
from matplotlib import pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.indicators.igd import IGD
from pymoo.operators.sampling.lhs import sampling_lhs
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import Hypervolume

from DTLZ_problem import DTLZbProblem, WFGcProblem, get_custom_problem
from DTLZ_problem import evaluate, get_pf, get_moea_data, get_ps
from DTLZ_problem import DTLZ_PROBLEM_NAMES
from benchmarking import benchmark_for_seeds
from maml_mod import MamlWrapperNaive as MamlWrapper
from problem_config.example import get_args, get_network_structure, get_dataset
from utils import NamedDict, set_ipython_exception_hook
from visualization import visualize_pf, visualize_igd


def cprint(*args, do_print=True, **kwargs):
    """
    Conditional print wrapper
    """
    if do_print:
        print(*args, **kwargs)


def generate_dataset(additional_data, args, dataset_problem_list, problem_name, dim, n_var) -> tuple:
    """
    Inner function to generate dataset

    Returns
    -------
    tuple of (dataset_x, dataset, min_max, delta)
    """
    if additional_data is None:
        delta = []
        for i in range(2):
            delta.append([np.random.rand(args.train_test[i]) * 30, np.random.rand(args.train_test[i]) * 30])
        # dataset_x = [train_support_x, train_query_x, test_support_x, test_query_x]
        dataset_x = [None, None, None, None]
        # generate test_support_x for both surrogate and baseline moea
        dataset_x[2] = sampling_lhs(n_samples=1000, n_var=n_var, xl=0, xu=1)
        dataset_x[2] = dataset_x[2][np.random.choice(dataset_x[2].shape[0], args.k_spt, replace=False), :]
        dataset, min_max = get_dataset(
            args,
            normalize_targets=True,
            delta=delta,
            problem_name=dataset_problem_list,
            test_problem_name=[problem_name],
            pf_ratio=0,
            dim=dim
        )
    else:
        delta = additional_data['delta']
        dataset_x = additional_data['dataset_x']
        dataset = additional_data['dataset']
        min_max = additional_data['min_max']
    return dataset_x, dataset, min_max, delta


def get_plot_scale(target_pf, true_pf, n_objectives):
    scale = []
    for i in range(n_objectives):
        concatenated = np.concatenate([target_pf[:, i], true_pf[:, i]]),
        data = [np.min(concatenated), np.max(concatenated)]
        scale.append(data)
    return scale


def main(problem_name: str,
         dataset_problem_list: List[str],
         selection_method: str,
         print_progress=False,
         do_plot=False,
         do_train=True,
         gpu_id: int | None = None,
         return_none_train_igd=False,
         additional_data: dict | None = None):
    """
    Main function to run the benchmark

    Parameters
    ----------
    problem_name : str
        Name of the problem to perform benchmark on
    dataset_problem_list : List[str]
        List of problem names to use as data
    selection_method: str
        Selection method to use
    print_progress : bool
        Whether to print progress
    do_plot : bool
        Whether to plot the result (IGDs, PFs)
    do_train : bool
        Whether to perform meta-train
    gpu_id : int | None
        GPU ID to use, if None, use the device in @get_args()
    return_none_train_igd : bool
        If to use as an intermediate function to
         return IGD of our method without meta-train
    additional_data : dict | None
        Additional data to use, if None, generate new data
         require not None if @return_none_train_igd is True
    """

    # serve as an internal function call, suppress output
    if return_none_train_igd:
        print_progress = False
        do_plot = False
        do_train = False

    ############################
    ## Define Hyper-Parameters #
    ############################
    args = get_args()
    dim = args.dim if 'dim' in args else 0
    if gpu_id is not None:
        args.device = torch.device(f'cuda:{gpu_id}')
    n_var = args.problem_dim[0]
    n_objectives = args.problem_dim[1]

    # initial function evaluation number
    fn_eval = args.k_spt
    fn_eval_limit = 300 + 2

    # max number of new individuals to add
    # to the training set of the surrogate model
    max_pts_num = 20
    moea_pop_size = 50
    proxy_n_gen = 100
    proxy_pop_size = 100
    network_structure = get_network_structure(args)

    # dataset related
    delta: list
    dataset_x: list
    dataset: tuple
    min_max: tuple

    # dataset generation
    res = generate_dataset(additional_data, args, dataset_problem_list, problem_name, dim, n_var)
    dataset_x, dataset, min_max, delta = res

    # the delta and the initial dataset_x for the problem
    problem_delta = np.array(delta[1])[:, -1]
    init_x = dataset[1][0][0]

    # the (true) Pareto front of the problem,
    # acquired by MOEA with adequate
    # amount of function evaluations
    problem_pf: np.ndarray
    problem_ps: np.ndarray

    igd_metric: IGD

    ##############################
    ## Meta Model Initialization #
    ##############################
    meta = MamlWrapper(dataset, args, network_structure)
    cprint('dataset init complete', do_print=print_progress)
    if do_train:
        train_loss = meta.train(explicit=print_progress)
        cprint(f'train_loss: {train_loss[-1]}', do_print=print_progress)
    test_loss = meta.test(return_single_loss=False)
    cprint('MAML init complete', do_print=print_progress)

    #######################
    ## Initialize Problem #
    #######################
    problem = get_custom_problem(name=problem_name,
                                 n_var=n_var,
                                 n_obj=n_objectives,
                                 delta1=problem_delta[0],
                                 delta2=problem_delta[1])

    if additional_data is None:
        problem_pf = get_pf(n_objectives, problem, min_max)
        problem_ps = get_ps(n_var, n_objectives, problem_delta[0], problem_delta[1], problem_name)
    else:
        problem_pf = additional_data['problem_pf']
        problem_ps = additional_data['problem_ps']

    # serve as a simpler way of performing non-dominated sorting
    res = minimize(problem=problem,
                   algorithm=NSGA2(pop_size=proxy_pop_size, sampling=init_x),
                   termination=('n_gen', 0.1))

    history_x, history_f = res.X, res.F
    history_x = history_x.astype(np.float32)
    history_f = history_f.astype(np.float32)
    if min_max[0] is not None:
        history_f -= min_max[0]
        history_f /= min_max[1]

    igd_metric = IGD(problem_pf, zero_to_one=True)

    #####################
    ## IGDs Declaration #
    #####################

    # The IGD of our method, i.e., Meta-Guided Surrogate-Assisted MOEA
    ours_igd = [igd_metric.do(history_f)] * 2
    # The IGD of the Surrogate Model as an approximation of the real problem
    # For visualization purpose
    surrogate_per_update_idg = [*ours_igd]
    # The IGD of the Baseline MOEA
    moea_igd: list | np.ndarray
    # The IGD of our method, only that it is *NOT* meta-trained
    ours_no_meta_igd: list | np.ndarray

    # igd indexes
    ours_igd_index = [0, fn_eval]  # for ours_igd and surrogate_per_update_idg
    moea_igd_index: list | np.ndarray
    ours_no_meta_igd_index: list | np.ndarray

    cprint('Algorithm init complete', do_print=print_progress)

    ################################
    ## Start Fine-Tuning Surrogate #
    ################################
    # parameters for plot the PF of the *surrogate model*
    plot_interval = 80
    plotted = 1
    while fn_eval < fn_eval_limit:
        cprint(f'fn_eval: {fn_eval}', do_print=print_progress)

        ########################################
        ## Calculate PF of the Surrogate Model #
        ########################################
        ref_dirs = get_reference_directions("das-dennis", n_objectives, n_partitions=8)
        algorithm_surrogate = RVEA(pop_size=proxy_pop_size, sampling=history_x, ref_dirs=ref_dirs)
        problem_surrogate = DTLZbProblem(n_var=n_var, n_obj=n_objectives, sol=meta) if problem_name[0] == 'D' \
            else WFGcProblem(n_var=n_var, n_obj=n_objectives, sol=meta)
        res = minimize(problem_surrogate,
                       algorithm_surrogate,
                       ('n_gen', proxy_n_gen),
                       verbose=False)

        # the objective of the real problem by the Pareto set of the surrogate model
        # For visualization purpose
        if do_plot:
            objective_true = evaluate(res.X, problem_delta, n_objectives, problem_name, min_max=min_max)
            surrogate_per_update_idg.append(igd_metric.do(objective_true))

        # select individuals to add to the training set
        if selection_method == 'ns':
            surrogate_pareto_set = res.X
            if len(surrogate_pareto_set) > max_pts_num:
                surrogate_pareto_set = surrogate_pareto_set[np.random.choice(surrogate_pareto_set.shape[0], max_pts_num)]
            eval_x = surrogate_pareto_set.astype(np.float32)
        elif selection_method == 'hv':
            surrogate_pop = res.pop
            surrogate_f = np.array([ind.F for ind in surrogate_pop])
            approx_ideal = surrogate_f.min(axis=0)
            approx_nadir = surrogate_f.max(axis=0)
            metric = Hypervolume(ref_point= np.array([1.1, 1.1, 1.1]),
                     norm_ref_point=False,
                     zero_to_one=True,
                     ideal=approx_ideal,
                     nadir=approx_nadir)
            surrogate_X = [ind.X for ind in surrogate_pop]
            surrogate_hv = []
            for ind in surrogate_pop:
                surrogate_hv.append(metric.do(ind.F))
            zip_arr = zip(surrogate_hv,surrogate_X)
            sorted_zip = sorted(zip_arr, key=lambda x:x[0], reverse=True)
            sorted_hv, sorted_X = zip(*sorted_zip)
            eval_x = np.array(sorted_X[:max_pts_num]).astype(np.float32)
        
        eval_y = evaluate(eval_x, problem_delta, n_objectives, problem_name, min_max=min_max)
        eval_y = eval_y.astype(np.float32)

        fn_eval += eval_x.shape[0]
        history_x = np.vstack((history_x, eval_x))
        history_f = np.vstack((history_f, eval_y))

        ##################################################
        # Fine-Tune the Surrogate Model With New Dataset #
        ##################################################
        cont_loss = 0  # suppress ide warning of (possibly) undefined variable
        for _ in range(5):
            cont_loss = meta.test_continue(history_x, history_f, return_single_loss=True)
        cprint(f'continue loss: {cont_loss}', do_print=print_progress)

        ours_igd.append(igd_metric.do(history_f))
        ours_igd_index.append(fn_eval)

        # plot the PF of the surrogate model
        if fn_eval > plotted + plot_interval and do_plot:
            # plot the PF of the surrogate model
            surrogate_pf = get_pf(n_objectives, problem_surrogate)
            scale = get_plot_scale(surrogate_pf, problem_pf, n_objectives)
            plotted = fn_eval
            visualize_pf(pf=surrogate_pf, label='Surrogate PF', color='green',
                         scale=scale, pf_true=problem_pf)
            # plot the PS evaluated by the surrogate model
            surrogate_pf = problem_surrogate.evaluate(problem_ps, {})
            scale = get_plot_scale(surrogate_pf, problem_pf, n_objectives)
            plotted = fn_eval
            visualize_pf(pf=surrogate_pf, label='PS evaluate by surrogate', color='magenta',
                         scale=scale, pf_true=problem_pf, show=True)

    if return_none_train_igd:
        return ours_igd_index, ours_igd

    cprint('Algorithm complete', do_print=print_progress)

    ref_dirs = get_reference_directions("das-dennis", n_objectives, n_partitions=8)
    moea_problem = RVEA(pop_size=moea_pop_size, ref_dirs=ref_dirs)
    moea_pf, moea_igd_index, moea_igd = get_moea_data(n_var, n_objectives, problem_delta,
                                                    moea_problem,
                                                    fn_eval_limit,
                                                    igd_metric,
                                                    problem_name,
                                                    min_max)
    moea_igd_index = np.insert(moea_igd_index, 0, 0)
    moea_igd = np.insert(moea_igd, 0, ours_igd[0])
    cprint('MOEA Baseline complete', do_print=print_progress)

    if do_plot:
        # plot the PF of our method
        scale = get_plot_scale(history_f, problem_pf, n_objectives)
        visualize_pf(pf=history_f, label='Surrogate PF', color='green', scale=scale, pf_true=problem_pf)

        # plot the PF acquired by MOEA
        scale = get_plot_scale(moea_pf, problem_pf, n_objectives)
        visualize_pf(pf=moea_pf, label='MOEA PF', color='blue', scale=scale, pf_true=problem_pf)

        # get the IGD of our method without meta-training
        additional_data = {
            'delta': delta,
            'dataset_x': dataset_x,
            'dataset': dataset,
            'min_max': min_max,
            'problem_pf': problem_pf,
            'problem_ps': problem_ps
        }
        ours_no_meta_igd_index, ours_no_meta_igd = main(problem_name=problem_name,
                                                        dataset_problem_list=dataset_problem_list,
                                                        selection_method=selection_method,
                                                        return_none_train_igd=True,
                                                        additional_data=additional_data)

        # plot the IGDs
        plot_index_list = [ours_igd_index, moea_igd_index, ours_igd_index, ours_no_meta_igd_index]
        igd_list        = [ours_igd, moea_igd, surrogate_per_update_idg, ours_no_meta_igd]
        color_list      = ['black', 'blue', 'green', 'orange']
        label_list      = ['Our Algorithm with Meta', 'MOEA', 'Surrogate IGD per update', 'Our Algorithm without Meta']
        visualize_igd(plot_index_list, igd_list, color_list, label_list)
        plt.show()

    cprint(f'IGD of Proxy: {ours_igd[-2:]}', do_print=print_progress)
    cprint(f'IGD of MOEA:  {moea_igd[-2:]}', do_print=print_progress)

    # deallocate memory
    del meta
    return ours_igd[-1], moea_igd[-1]


def post_mean_std(data: list | np.ndarray):
    data = np.array(data)
    return np.mean(data, axis=0), np.std(data, axis=0)


def main_benchmark(problem_name: str):
    _seeds = 20
    _n_proc = 20
    init_seed = 42
    _estimate_gram = 3.5
    gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    data_problem_list = [DTLZ_PROBLEM_NAMES.d2c, DTLZ_PROBLEM_NAMES.d3c, DTLZ_PROBLEM_NAMES.d4c]
    _res = benchmark_for_seeds(main,
                               post_mean_std,
                               seeds=_seeds,
                               func_args=[problem_name, data_problem_list],
                               func_kwargs={'print_progress': False, 'do_train': True},
                               n_proc=_n_proc,
                               gpu_ids=gpu_ids,
                               estimated_gram=_estimate_gram,
                               init_seed=init_seed)
    print(f'MAML All IGD: {_res[0][0]} +- {_res[1][0]}')
    print(f'MOEA IGD:     {_res[0][1]} +- {_res[1][1]}')
    _res = benchmark_for_seeds(main,
                               post_mean_std,
                               seeds=_seeds,
                               func_args=[problem_name, data_problem_list],
                               func_kwargs={'print_progress': False, 'do_train': False},
                               n_proc=_n_proc,
                               gpu_ids=gpu_ids,
                               estimated_gram=_estimate_gram,
                               init_seed=init_seed)
    print(f'MAML One IGD: {_res[0][0]} +- {_res[1][0]}')


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
    set_ipython_exception_hook()
    fast_seed(20010924)

    _data_problem_list = [DTLZ_PROBLEM_NAMES.d4c]
    main(problem_name=DTLZ_PROBLEM_NAMES.d4c,
         dataset_problem_list=_data_problem_list,
         selection_method='hv',
         do_plot=False,
         print_progress=True,
         do_train=True
         )
