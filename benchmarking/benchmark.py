import gc
import os
import random
import warnings
from typing import Callable, List, Any, Literal
from multiprocessing import Pool, set_start_method, get_start_method


def _multi_processing_wrapper(data: tuple) -> Any:
    """
    Wrapper for multiprocessing

    Parameters
    ----------
    data : tuple
        The data to be passed to the function;
        The first 5 elements are required, as follows:
        1. The function to be called
        2. The arguments of the function
        3. The keyword arguments of the function
        4. The seed to be used
        5. The availability of the libraries
        The rest will be ignored

    Returns
    -------
    Any
        The result of the function
    """
    func, args, kwargs, seed, availabilities, *_ = data
    if availabilities['numpy']:
        import numpy as np
        np.random.seed(seed)
    # if availabilities['scipy']:
    #     import scipy
    #     scipy.random.seed(seed)
    if availabilities['torch']:
        import torch
        torch.manual_seed(seed)
    if availabilities['tensorflow']:
        import tensorflow as tf
        tf.random.set_seed(seed)
    print(f'{seed=}')

    try:
        ret = func(*args, **kwargs)
    except Exception as e:
        print(f'Error in {seed=}: {e}', flush=True)
        ret = None

    print(f'{seed=} finished')

    # deallocate the memory
    gc.collect()
    if availabilities['torch']:
        import torch
        torch.cuda.empty_cache()
    if availabilities['tensorflow']:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    return ret


def benchmark_for_seeds(func: Callable,
                        post_process: Callable,
                        seeds: List[int] | int = 10,
                        func_args: List[Any] = None,
                        func_kwargs: dict | None = None,
                        post_process_args: List[Any] = None,
                        post_process_kwargs: dict | None = None,
                        n_proc: int = 1,
                        new_proc_method: Literal['fork', 'spawn'] = 'spawn',
                        init_seed: int = 42) -> Any:
    """
    Benchmark the function for different seeds

    Parameters
    ----------
    func : Callable
        The function to be benchmarked
    post_process : Callable
        The post process function
    seeds : List[int] | int
        The seeds to be used
    func_args : List[Any]
        The arguments of the function
    func_kwargs : dict | None
        The keyword arguments for the function
    post_process_args : List[Any]
        The arguments of the post process function
    post_process_kwargs : dict | None
        The keyword arguments for the post process function
    n_proc : int
        The number of processes to be used
        If n_proc <= 0, then the number of processes will be set to the number of available CPUs
        If n_proc == 1, then the multiprocessing will be disabled, useful if function cannot be pickled
        If n_proc > 1, then the multiprocessing will be enabled for `n_proc` processes
    new_proc_method : Literal['fork', 'spawn']
        The method to be used to create new processes,
        'fork' only works on Unix-like systems, while 'spawn' works on all platforms
        If using CUDA, 'spawn' is recommended
    init_seed : int = 42
        The initial seed to be used to generate the seeds if `seeds` is an integer

    Returns
    -------
    Any
        The result of the post process function
    """
    # initialize the random seeds with 42
    if get_start_method() != new_proc_method:
        try:
            set_start_method(new_proc_method)
        except RuntimeError as e:
            pass
    random.seed(init_seed)
    availabilities = {
        'numpy': False,
        # 'scipy': False,
        'torch': False,
        'tensorflow': False,
    }
    try:
        import numpy as np
        availabilities['numpy'] = True
    except ImportError:
        pass
    # try:
    #     import scipy
    #     availabilities['scipy'] = True
    # except ImportError:
    #     pass
    try:
        import torch
        availabilities['torch'] = True
    except ImportError:
        pass
    try:
        import tensorflow as tf
        availabilities['tensorflow'] = True
    except ImportError:
        pass
    if func_args is None:
        func_args = []
    if func_kwargs is None:
        func_kwargs = {}
    if post_process_args is None:
        post_process_args = []
    if post_process_kwargs is None:
        post_process_kwargs = {}
    if isinstance(seeds, int):
        seeds = random.sample(range(100000), seeds)
    if n_proc <= 0:
        n_proc = os.cpu_count()
        if n_proc is None:
            n_proc = 1
    results = []
    if n_proc > 1:
        with Pool(n_proc, maxtasksperchild=1) as pool:
            results = pool.map(_multi_processing_wrapper, [
                (func, func_args, func_kwargs, seed, availabilities)
                for seed in seeds
            ])
    else:
        for seed in seeds:
            results.append(_multi_processing_wrapper((func, func_args, func_kwargs, seed, availabilities)))
    ret = post_process(results, *post_process_args, **post_process_kwargs)
    gc.collect()
    return ret


def benchmark_for_seeds_different_args(func: Callable,
                                       post_process: Callable,
                                       seeds: List[int] | int = 10,
                                       func_args: List[List[Any]] = None,
                                       func_kwargs: List[dict] | None = None,
                                       post_process_args: List[Any] = None,
                                       post_process_kwargs: dict | None = None,
                                       n_proc: int = 1,
                                       new_proc_method: Literal['fork', 'spawn'] = 'spawn',
                                       init_seed: int = 42) -> Any:
    """
    See `benchmark_for_seeds` for the meaning of the parameters, except that args and kwargs are lists
    """
    # initialize the random seeds with 42
    if get_start_method() != new_proc_method:
        try:
            set_start_method(new_proc_method)
        except RuntimeError as e:
            pass
    random.seed(init_seed)
    availabilities = {
        'numpy': False,
        # 'scipy': False,
        'torch': False,
        'tensorflow': False,
    }
    try:
        import numpy as np
        availabilities['numpy'] = True
    except ImportError:
        pass
    # try:
    #     import scipy
    #     availabilities['scipy'] = True
    # except ImportError:
    #     pass
    try:
        import torch
        availabilities['torch'] = True
    except ImportError:
        pass
    try:
        import tensorflow as tf
        availabilities['tensorflow'] = True
    except ImportError:
        pass
    if func_args is None:
        func_args = [[]] * len(seeds)
    if func_kwargs is None:
        func_kwargs = [{}] * len(seeds)
    if post_process_args is None:
        post_process_args = []
    if post_process_kwargs is None:
        post_process_kwargs = {}
    if isinstance(seeds, int):
        seeds = random.sample(range(100000), seeds)
    if n_proc <= 0:
        n_proc = os.cpu_count()
        if n_proc is None:
            n_proc = 1
    results = []
    if n_proc > 1:
        with Pool(n_proc, maxtasksperchild=1) as pool:
            results = pool.map(_multi_processing_wrapper, [
                (func, func_args[i], func_kwargs[i], seed, availabilities)
                for i, seed in enumerate(seeds)
            ])
    else:
        for i, seed in enumerate(seeds):
            results.append(_multi_processing_wrapper((func, func_args[i], func_kwargs[i], seed, availabilities)))
    ret = post_process(results, *post_process_args, **post_process_kwargs)
    gc.collect()
    return ret
