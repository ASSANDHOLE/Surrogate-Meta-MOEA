import torch

from DTLZ_problem import create_dataset

from utils import NamedDict


def get_args():
    args = NamedDict()
    args.problem_dim = (8, 3)
    args.train_test = (20, 3)
    args.epoch = 100
    args.update_lr = 0.01
    args.meta_lr = 0.001
    args.k_spt = 20
    args.k_qry = 100
    args.update_step = 5
    args.update_step_test = 10
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.device = torch.device('cpu')
    return args


def get_network_structure(args):
    n_args = args.problem_dim[0]
    config = [
        ('linear', [2*n_args, n_args]),
        ('leakyrelu', [0.1, True]),
        ('linear', [2*n_args, 2*n_args]),
        ('leakyrelu', [0.1, True]),
        ('linear', [1, 2*n_args]),
    ]
    return config


def get_dataset(args, **kwargs):
    problem_dim = args.problem_dim
    n_problem = args.train_test
    spt_qry = (args.k_spt, args.k_qry)
    dataset = create_dataset(problem_dim, n_problem=n_problem, spt_qry=spt_qry, **kwargs)
    return dataset


