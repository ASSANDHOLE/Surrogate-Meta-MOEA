import sys

sys.path.append('..')

import platform

import torch

from DTLZ_problem import create_dataset

from utils import NamedDict


def get_device(force_device: str = None):
    if force_device is not None:
        return torch.device(force_device)

    if platform.system().lower() == 'darwin' and \
            platform.processor().lower().startswith('arm'):
        has_mps_support = False
        try:
            has_mps_support = torch.backends.mps.is_available()
        except AttributeError:
            pass
        # Enable MPS support for Apple Silicon
        # See https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/
        device = torch.device('mps' if has_mps_support else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def get_args():
    args = NamedDict()
    args.dim = 0
    args.problem_dim = (10, 3)
    args.train_test = (1000, 1)
    args.epoch = 10
    args.sgd_epoch = 10
    args.sgd_select_n = 100
    args.update_lr = 0.02
    args.meta_lr = 0.01
    args.fine_tune_lr = 0.03
    args.k_spt = 150
    args.k_qry = 600
    args.update_step = 10
    args.update_step_test = 15
    args.device = get_device()
    # args.device = torch.device('cpu')
    return args


def get_network_structure(args):
    n_args = args.problem_dim[0]
    n_args_out = args.problem_dim[1]
    if 'dim' in args:
        n_args_out = int(n_args_out ** args.dim)
    config = [
        ('linear', [100, n_args]),
        ('relu', [True]),
        ('linear', [100, 100]),
        ('relu', [True]),
        # ('linear', [200, 200]),
        # ('relu', [True]),
        # ('linear', [200, 200]),
        # ('relu', [True]),
        # ('linear', [30, 60]),
        # ('relu', [True]),
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
