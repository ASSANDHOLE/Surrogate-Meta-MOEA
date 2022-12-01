import sys

sys.path.append('..')

import torch

from DTLZ_problem import create_dataset

from utils import NamedDict


def get_args():
    args = NamedDict()
    args.dim = 1
    args.problem_dim = (10, 3)
    args.train_test = (300, 1)
    args.epoch = 10
    args.sgd_epoch = 10
    args.sgd_select_n = 50
    args.update_lr = 0.0025
    args.meta_lr = 0.001
    args.fine_tune_lr = 0.005
    args.k_spt = 30
    args.k_qry = 200
    args.update_step = 10
    args.update_step_test = 15
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.device = torch.device('cpu')
    return args


def get_network_structure(args):
    n_args = args.problem_dim[0]
    n_args_out = args.problem_dim[1]
    if 'dim' in args:
        n_args_out = int(n_args_out ** args.dim)
    config = [
        # ('linear', [2 * n_args, n_args]),
        # ('relu', [True]),
        # ('linear', [4 * n_args, 2 * n_args]),
        # ('relu', [True]),
        # ('linear', [4 * n_args, 4 * n_args]),
        # ('relu', [True]),
        # ('linear', [2 * n_args, 4 * n_args]),
        # ('relu', [True]),
        # ('linear', [1, 2 * n_args]),
        ('linear', [100, n_args]),
        ('relu', [True]),
        ('linear', [200, 100]),
        ('relu', [True]),
        ('linear', [200, 200]),
        ('relu', [True]),
        ('linear', [200, 200]),
        ('relu', [True]),
        ('linear', [100, 200]),
        ('relu', [True]),
        ('linear', [n_args_out, 100]),
    ]
    return config


def get_dataset(args, **kwargs):
    problem_dim = args.problem_dim
    n_problem = args.train_test
    spt_qry = (args.k_spt, args.k_qry)
    dataset = create_dataset(problem_dim, n_problem=n_problem, spt_qry=spt_qry, **kwargs)
    return dataset


def estimate_resource_usage():
    usage = NamedDict()
    usage.memory = '2.3G'
    usage.gpu_memory = '1077M'
    usage.description = \
        "This is the estimate resource usage for this setup.\n" \
        "Test platform:\n" \
        "    OS:      Ubuntu 22.04 LTS\n" \
        "    CPU:     Intel(R) Core(TM) i9-10900K CPU\n" \
        "    GPU:     NVIDIA GeForce RTX 3090\n" \
        "    Memory:  32G\n"
    return usage
