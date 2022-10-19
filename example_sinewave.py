import torch

from sine_dataset import create_dataset_sinewave

from utils import NamedDict


def get_args_maml_regression():
    args = NamedDict()
    args.problem_dim = (1, 3)
    args.train_test = (20, 3)
    args.epoch = 100
    args.update_lr = 0.00001  # with other configs unchanged, lr=0.01 causes loss=nan, i.e., gradient explode
    args.meta_lr = 0.001
    args.k_spt = 20
    args.k_qry = 100
    args.update_step = 5
    args.update_step_test = 10
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.device = torch.device('cpu')
    return args


def get_network_structure_maml_regression():
    config = [
        ('linear', [40, 1]),
        ('leakyrelu', [0.1, True]),
        ('linear', [40, 40]),
        ('leakyrelu', [0.1, True]),
        ('linear', [1, 40]),
    ]
    return config


def get_dataset_sinewave(args, **kwargs):
    problem_dim = args.problem_dim
    n_problem = args.train_test
    spt_qry = (args.k_spt, args.k_qry)
    dataset = create_dataset_sinewave(problem_dim, n_problem, spt_qry=spt_qry, **kwargs)
    return dataset


