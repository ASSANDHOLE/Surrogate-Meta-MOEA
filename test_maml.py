import math
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from maml_mod import Meta

from utils import NamedDict

N_ARG_IN = 8
N_ARG_OUT = 3
N_TEST = 1


def get_network_structure():
    """
    See all supported layers in `Learner.forward()` in `maml_mod/learner.py`
    """
    config = [
        ('linear', [2*N_ARG_IN, N_ARG_IN]),
        ('relu', [True]),
        ('linear', [2*N_ARG_IN, 2*N_ARG_IN]),
        ('relu', [True]),
        ('linear', [N_ARG_IN, 2*N_ARG_IN]),
        ('relu', [True]),
        ('linear', [1, N_ARG_IN]),
    ]
    return config


class Baseline(torch.nn.Module):
    """
    * FOR DEMONSTRATION PURPOSES ONLY *
    The equivalent PyTorch Module impl of the `get_network_structure`
    """
    def __init__(self, input_size, output_size):
        super(Baseline, self).__init__()
        self.fcs = torch.nn.Sequential(
            torch.nn.Linear(input_size, 2*input_size),
            torch.nn.ReLU(),
            torch.nn.Linear(2*input_size, 2*input_size),
            torch.nn.ReLU(),
            torch.nn.Linear(2*input_size, input_size),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size, output_size),
        )

    def forward(self, x):
        out = self.fcs(x)
        return out


def get_args():
    args = NamedDict()
    args.epoch = 50
    args.update_lr = 0.01
    args.meta_lr = 0.001
    args.n_way = N_ARG_OUT + N_TEST
    args.k_spt = 20
    args.k_qry = 100
    args.task_num = N_ARG_OUT
    args.update_step = 10
    args.update_step_test = 20
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.device = torch.device('cpu')
    return args


def test_dataset(n_train: int, n_test: int) -> Tuple[List[list], List[list]]:
    def noise(n: int) -> np.ndarray:
        return np.random.normal(0, 1, n)

    def f1(x: np.ndarray) -> float:
        sin_val = np.sin(x).sum()
        ratio = [math.sin(x) for x in range(N_ARG_IN)]
        linear_x = [x ** i for i, x in enumerate(x, 1)]
        return sin_val + np.dot(ratio, linear_x) + noise(1)[0]

    def f2(x: np.ndarray) -> float:
        left_half = x[:N_ARG_IN // 2]
        right_half = x[N_ARG_IN // 2:]
        left_half = np.array([lh ** 2 for lh in left_half])
        right_half = np.array([rh ** 0.5 for rh in right_half])
        return np.dot(left_half, right_half) + noise(1)[0]

    def f3(x: np.ndarray) -> float:
        return np.sum(x) + noise(1)[0]

    def f_test(x: np.ndarray) -> float:
        return (f1(x) + f2(x) + f3(x)) / 3 + noise(1)[0]

    def random_x():
        return np.random.random(N_ARG_IN)

    train_set, test_set = [[] for _ in range(N_ARG_OUT + 1)], [[] for _ in range(N_ARG_OUT + 1)]

    fs = [f1, f2, f3, f_test]

    def append_data(data, n):
        for _ in range(n):
            for i in range(N_ARG_OUT + 1):
                rx = random_x()
                data[i].append((rx, fs[i](rx)))

    append_data(train_set, n_train)
    append_data(test_set, n_test)

    return train_set, test_set


def draw_result(x, y_true, y_maml, y_baseline):
    if N_ARG_IN != 1:
        x = x[:, 0]
    plt.plot(x, y_true, 'r', label='true')
    plt.plot(x, y_maml, 'b', label='maml')
    plt.plot(x, y_baseline, 'g', label='baseline')
    plt.legend()
    plt.show()


def main():
    args = get_args()
    dev = args.device
    net_structure = get_network_structure()
    train_set, test_set = test_dataset(args.k_spt, args.k_qry)
    maml = Meta(args, net_structure).to(dev)
    rand_maml = Meta(args, net_structure).to(dev)

    def get_train_test_xy(data):
        train_x = np.array([[data[i][j][0] for j in range(len(data[i]))] for i in range(N_ARG_OUT)])
        train_y = np.array([[data[i][j][1] for j in range(len(data[i]))] for i in range(N_ARG_OUT)])
        test_x = np.array([data[-1][j][0] for j in range(len(data[-1]))])
        test_y = np.array([data[-1][j][1] for j in range(len(data[-1]))])
        train_x = train_x.astype(np.float32)
        train_y = train_y.astype(np.float32)
        test_x = test_x.astype(np.float32)
        test_y = test_y.astype(np.float32)
        return train_x, train_y, test_x, test_y

    # train_x_spt.shape = (<train_task_num>, args.k_spt, N_ARG_IN)
    # train_y_spt.shape = (<train_task_num>, args.k_spt)
    # test_x_spt.shape = (<test_task_num>, args.k_spt, N_ARG_IN)
    # test_y_spt.shape = (<test_task_num>, args.k_spt)
    train_x_spt, train_y_spt, test_x_spt, test_y_spt = get_train_test_xy(train_set)

    # train_x_qry.shape = (<train_task_num>, args.k_qry, N_ARG_IN)
    # ...
    train_x_qry, train_y_qry, test_x_qry, test_y_qry = get_train_test_xy(test_set)

    def to_ts(x):
        return torch.from_numpy(x).to(dev)

    train_x_spt, train_y_spt, test_x_spt, test_y_spt = to_ts(train_x_spt), to_ts(train_y_spt), to_ts(test_x_spt), to_ts(
        test_y_spt)
    train_x_qry, train_y_qry, test_x_qry, test_y_qry = to_ts(train_x_qry), to_ts(train_y_qry), to_ts(test_x_qry), to_ts(
        test_y_qry)

    for ep in range(args.epoch):
        loss = maml(train_x_spt, train_y_spt, train_x_qry, train_y_qry)
        print('epoch: {}, train loss: {}'.format(ep, loss))

    maml_loss, res_maml = maml.fine_tuning(test_x_spt, test_y_spt, test_x_qry, test_y_qry)
    print('test loss: {}'.format(maml_loss))

    # baseline example

    baseline = Baseline(N_ARG_IN, 1).to(dev)
    baseline_opt = torch.optim.Adam(baseline.parameters(), lr=args.update_lr)
    for ep in range(args.update_step_test * args.epoch):
        baseline_opt.zero_grad()
        res_baseline = baseline(test_x_spt)
        loss = torch.nn.functional.mse_loss(res_baseline, test_y_spt.unsqueeze(1))
        loss.backward()
        baseline_opt.step()
    res_baseline = baseline(test_x_qry)
    baseline_loss = torch.nn.functional.mse_loss(res_baseline, test_y_qry.unsqueeze(1))
    print('baseline test loss: {}'.format(baseline_loss))
    # draw_result(test_x_qry.detach().cpu().numpy(), test_y_qry.detach().cpu().numpy(),
    #             res_maml.detach().cpu().numpy(), res_baseline.detach().cpu().numpy())
    rand_loss, rand_maml_result = rand_maml.fine_tuning(test_x_spt, test_y_spt, test_x_qry, test_y_qry)
    print('random maml test loss: {}'.format(rand_loss))
    return baseline_loss.item(), maml_loss, rand_loss


if __name__ == '__main__':
    TOT_RUN = 10
    loss_list = [[], [], []]
    print_fn = print

    # suppress output during training
    print = lambda *args, **kwargs: None

    for _ in range(TOT_RUN):
        baseline_loss, maml_loss, rand_loss = main()
        loss_list[0].append(baseline_loss)
        loss_list[1].append(maml_loss)
        loss_list[2].append(rand_loss)

    print = print_fn
    # baseline loss not printed (calculated differently)
    print(f'maml loss: {np.mean(loss_list[1])}')
    print(f'rand loss: {np.mean(loss_list[2])}')
