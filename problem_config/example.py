import sys

sys.path.append('..')

import torch

from DTLZ_problem import create_dataset

from utils import NamedDict


def get_args():
    args = NamedDict()
    args.problem_dim = (10, 3)
    args.train_test = (300, 1)
    args.epoch = 100
    args.update_lr = 0.02
    args.meta_lr = 0.01
    args.fine_tune_lr = 0.05
    args.k_spt = 30
    args.k_qry = 200
    args.update_step = 20
    args.update_step_test = 25
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.device = torch.device('cpu')
    return args


def get_network_structure(args):
    n_args = args.problem_dim[0]
    config = [
        ('linear', [15, n_args]),
        ('relu', [True]),
        ('linear', [15, 15]),
        ('relu', [True]),
        ('linear', [15, 15]),
        ('relu', [True]),
        ('linear', [1, 15]),
        # ('linear', [100, n_args]),
        # ('relu', [True]),
        # ('linear', [200, 100]),
        # ('relu', [True]),
        # ('linear', [200, 200]),
        # ('relu', [True]),
        # ('linear', [100, 200]),
        # ('relu', [True]),
        # ('linear', [1, 100]),
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
