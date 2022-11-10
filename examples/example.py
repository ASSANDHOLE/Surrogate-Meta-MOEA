import sys

sys.path.append('..')

import torch

from DTLZ_problem import create_dataset

from utils import NamedDict


def get_args():
    args = NamedDict()
    args.problem_dim = (10, 3)
    args.train_test = (15, 1)
    args.epoch = 50
    args.update_lr = 0.01
    args.meta_lr = 0.001
    args.k_spt = 30
    args.k_qry = 200
    args.update_step = 5
    args.update_step_test = 30
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.device = torch.device('cpu')
    return args


def get_network_structure(args):
    n_args = args.problem_dim[0]
    config = [
        ('linear', [2 * n_args, n_args]),
        ('relu', [True]),
        ('linear', [4 * n_args, 2 * n_args]),
        ('relu', [True]),
        ('linear', [4 * n_args, 4 * n_args]),
        ('relu', [True]),
        ('linear', [2 * n_args, 4 * n_args]),
        ('relu', [True]),
        ('linear', [1, 2 * n_args]),
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
