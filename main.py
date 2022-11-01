# Unified MAML for this project
from __future__ import annotations
from cProfile import label

from typing import Tuple, List

import numpy as np
import torch

from maml_mod import Meta, Learner
from visualization import visualize_loss, visualize_pf, visualize_igd

from matplotlib import pyplot as plt

from DTLZ_problem import evaluate, get_pf, get_moea_data

from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.igd import IGD

from utils import NamedDict
from examples.example import get_args, get_network_structure, get_dataset
from examples.example_sinewave import get_args_maml_regression, get_network_structure_maml_regression, \
    get_dataset_sinewave


class Sol:
    def __init__(self,
                 dataset: Tuple[
                     Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                     Tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]
                 ],
                 args: NamedDict,
                 network_structure: List[Tuple[str, list | None]]
                 ) -> None:
        """
        MAML Wrapper class

        Parameters
        ----------
        dataset : Tuple[
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]
        ]
            The first element is the training set [support set, support label, query set, query label]
            The second element is the test set [support set, support label, query set, query label]

        args : NamedDict
            The arguments for the MAML model

        network_structure : List[Tuple[str, list | None]]
            The network structure of the MAML model, refer to `get_network_structure` in `test_maml.py` for example
        """

        self.train_set = dataset[0]
        self.test_set = dataset[1]
        self.args = args
        self.network_structure = network_structure
        if 'device' in self.args:
            self.device = self.args.device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.maml = Meta(self.args, self.network_structure).to(self.device)
        self._check_integrity()
        self.dateset_to_device()
        self.nets: List[Learner] = None
        self.fast_weights: List[list] = None

    def _check_integrity(self) -> None:
        """
        Check if the initial data is valid
        """
        assert len(self.train_set) == 4
        assert len(self.test_set) == 4
        required_attr = ['epoch', 'update_lr', 'meta_lr', 'k_spt',
                         'k_qry', 'update_step', 'update_step_test']
        assert all([hasattr(self.args, attr) for attr in required_attr])
        if 'n_way' not in self.args:
            self.args.n_way = 0
        if 'task_num' not in self.args:
            self.args.task_num = 0

    def dateset_to_device(self, device: torch.device = None) -> None:
        """
        Move the dataset to the device
        """
        device = device if device is not None else self.device
        self.train_set = tuple([torch.from_numpy(arr).to(device) for arr in self.train_set])
        self.test_set = tuple([torch.from_numpy(arr).to(device) for arr in self.test_set])

    def train(self, explicit: bool | int = False) -> List[float]:
        """
        Train the model

        Parameters
        ----------
        explicit : bool | int
            If explicit is True or a positive integer, the training loss will be print per {1 | explicit} epoch

        Returns
        -------
        List[float]:
            The training loss of each epoch
        """
        print_loss = explicit if isinstance(explicit, bool) else explicit > 0
        print_period = 1 if isinstance(explicit, bool) else explicit

        loss_arr = []
        for epoch in range(self.args.epoch):
            loss_arr.append(self.maml(*self.train_set))
            if print_loss and (epoch % print_period == 0 or epoch == self.args.epoch - 1):
                print(f'Epoch {epoch:4d}: {loss_arr[-1]:.4f}')
        return loss_arr

    def test(self, return_single_loss: bool = True, pretrain: bool = False) -> List[List[float] | float] | None:
        """
        Test the model

        Parameters
        ----------
        return_single_loss : bool
            If True, the loss of each gradient update will be returned
        pretrain : bool
            If True, pretrain the model using the train support set first

        Returns
        -------
        List[List[float] | float] | None:
            The loss of the model for each task, None if no test query set is provided
        """
        if not pretrain:
            loss, res, _nets, _fast_weights = self.maml.fine_tuning(*self.test_set,
                                                                    return_single_lose=return_single_loss)
            self.nets = _nets
            self.fast_weights = _fast_weights

        else:
            # train_x = torch.cat([self.train_set[0], self.train_set[2]], dim=1)
            # train_y = torch.cat([self.train_set[1], self.train_set[3]], dim=1)
            train_x = self.train_set[0]
            train_y = self.train_set[1]
            train_x = train_x.reshape((1, -1, train_x.shape[-1]))
            train_y = train_y.reshape((1, -1, 1))
            loss = self.maml.pretrain_fine_tuning(train_x, train_y, *self.test_set,
                                                  return_single_lose=return_single_loss)
        return loss

    def test_continue(self, x: np.ndarray, y: np.ndarray, return_single_loss: bool = True) -> None:
        """
        Test the model based on previous fine-tuned model

        Parameters
        ----------
        x : np.ndarray
            The new data for training
        y : np.ndarray
            The label of the new data
        return_single_loss : bool
            If True, the loss of each gradient update will be returned
        """
        if self.nets is None or self.fast_weights is None:
            raise ValueError('No previous fine-tuned model found, please use `test` instead')

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        x, y = x.to(self.device), y.to(self.device)

        _, _, _nets, _fast_weights = self.maml.fine_tuning_continue(self.nets, self.fast_weights, x, y, *((None,) * 4),
                                                                    return_single_lose=return_single_loss)
        self.nets = _nets
        self.fast_weights = _fast_weights

    def __call__(self, x: np.ndarray) -> List[float]:
        if self.nets is None or self.fast_weights is None:
            raise ValueError('No previous fine-tuned model found, please use `test` instead')
        x = torch.from_numpy(x).to(self.device)
        return [net(x, self.fast_weights[i]).detach().cpu().numpy().flatten()[0] for i, net in enumerate(self.nets)]


class MyProblem(Problem):
    def __init__(self, sol: Sol):
        self.sol = sol
        super().__init__(n_var=8,
                         n_obj=3,
                         #  n_constr=2,
                         xl=np.array([0] * 8, np.float32),
                         xu=np.array([1] * 8, np.float32),
                         )

    def _evaluate(self, x, out, *args, **kwargs):
        x = x.astype(np.float32)
        f = []
        for xi in x:
            fi = self.sol(xi)
            f.append(fi)
        out["F"] = np.array(f)


def main():
    # see Sol.__init__ for more information
    args = get_args()
    network_structure = get_network_structure(args)
    dataset, _ = get_dataset(args, normalize_targets=True)
    sol = Sol(dataset, args, network_structure)
    train_loss = sol.train(explicit=False)
    test_loss = sol.test(return_single_loss=False)
    mean_test_loss = np.mean(test_loss, axis=0)
    print(f'Test loss: {mean_test_loss[-1]:.4f}')
    x_test = np.array([i * 0.1 for i in range(1, 1 + 8)], np.float32)
    y_pred = sol(x_test)
    y_true = [y + 1 for y in y_pred]  # add some noise for testing
    sol.test_continue(np.array([x_test] * 3), np.array(y_true, np.float32).reshape((3, 1)))
    y_pred_1 = sol(x_test)
    print(f'Prediction: {y_pred}')
    print(f'Prediction after continue: {y_pred_1}')

    # args.update_step_test = int(1.5 * args.update_step_test)
    sol = Sol(dataset, args, network_structure)
    random_loss = sol.test(pretrain=True, return_single_loss=False)
    mean_random_loss = np.mean(random_loss, axis=0)
    print(f'Random loss: {mean_random_loss[-1]:.4f}')

    visualize_loss(test_loss, random_loss)


def main_sinewave():
    args = get_args_maml_regression()
    network_structure = get_network_structure_maml_regression()
    dataset = get_dataset_sinewave(args, normalize_targets=True)
    sol = Sol(dataset, args, network_structure)
    train_loss = sol.train(explicit=5)
    test_loss = sol.test(return_single_loss=False)
    mean_test_loss = np.mean(test_loss, axis=0)
    print(f'Test loss: {mean_test_loss[-1]:.4f}')

    args.update_step_test = int(1.5 * args.update_step_test)
    sol = Sol(dataset, args, network_structure)
    random_loss = sol.test(return_single_loss=False, pretrain=True)
    print(f'Random loss: {random_loss[-1]:.4f}')
    visualize_loss(mean_test_loss, random_loss)


def main_NSGA():
    args = get_args()
    network_structure = get_network_structure(args)
    # generate delta
    delta = []
    for i in range(2):
        delta.append([np.random.randint(0, 100, args.train_test[i]), np.random.randint(0, 10, args.train_test[i])])
    dataset, min_max = get_dataset(args, normalize_targets=True, delta=delta)
    sol = Sol(dataset, args, network_structure)
    # train_loss = sol.train(explicit=False)
    test_loss = sol.test(return_single_loss=False)

    n_var = args.problem_dim[0]
    n_objectives = args.problem_dim[1]
    delta_finetune = np.array(delta[1])[:, -1]

    pf_true = get_pf(n_var, n_objectives, delta_finetune, min_max)

    history_x, history_f = np.empty((0, n_var)), np.empty((0, n_objectives))
    igd = []
    x_size = []
    fn_eval_limit = 300
    max_pts_num = 5
    pop_size = 60
    sample_size = 30
    n_gen = 10

    while sum(x_size) < fn_eval_limit:

        if history_x.shape[0] < sample_size:
            random_x = np.random.rand(pop_size - history_x.shape[0], n_var)
            sample_x = np.vstack((history_x, random_x))
        else:
            random_x = np.random.rand(pop_size - sample_size, n_var)
            sample_x = np.vstack((history_x[np.random.choice(history_x.shape[0], sample_size)], random_x))

        algorithm = NSGA2(pop_size=pop_size, sampling=sample_x)

        res = minimize(MyProblem(sol=sol),
                       algorithm,
                       ("n_gen", n_gen),
                       seed=1,
                       verbose=False)
        
        X = res.X
        if len(X) > max_pts_num:
            X = X[np.random.choice(X.shape[0], max_pts_num)]
        
        history_x = np.vstack((history_x, X))
        history_f = np.vstack((history_f, res.F))

        x_size.append(X.shape[0])

        X = X.astype(np.float32)
        y_true = evaluate(X, delta_finetune, n_objectives, min_max=min_max)

        new_y_true = []
        for i in range(n_objectives):
            new_y_true.append(y_true[:, i])
        new_y_true = np.array(new_y_true, dtype=np.float32)
        new_y_true = new_y_true.reshape((*new_y_true.shape, 1))

        sol.test_continue(X, new_y_true)

        metric = IGD(pf_true, zero_to_one=True)
        igd.append(metric.do(history_f))
    
    pf = evaluate(res.X, delta_finetune, n_objectives, min_max=min_max)
    moea_pf, n_evals_moea, igd_moea = get_moea_data(n_var, n_objectives, delta_finetune, algorithm, int(fn_eval_limit/pop_size), min_max)
    n_evals_moea = n_evals_moea[:-1]
    igd_moea = igd_moea[:-1]

    visualize_pf(pf=pf, label='Sorrogate PF', color='green', scale=[0.5]*3, pf_true=pf_true)
    visualize_pf(pf=moea_pf, label='NSGA-II PF', color='blue', scale=[0.5]*3, pf_true=pf_true)

    func_evals = [max_pts_num*np.arange(len(igd)), n_evals_moea]
    igds = [igd, igd_moea]
    colors = ['black', 'red']
    labels = ["Our Surrogate Model", "NSGA-II"]
    visualize_igd(func_evals, igds, colors, labels)
    plt.show()


if __name__ == '__main__':
    # main()
    # main_sinewave()
    main_NSGA()

