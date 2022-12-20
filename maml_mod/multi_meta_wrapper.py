from typing import Tuple, List

import numpy as np
import torch

from maml_mod import Meta, Learner
from utils import NamedDict

from .wrapper_base import MamlWrapperAbc


class MamlWrapper(MamlWrapperAbc):
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
        self.train_sets = dataset[0]
        self.test_sets = dataset[1]
        self.args = args
        if 'dim' in args and args.dim != 0:
            raise ValueError('multi_meta_wrapper received a `dim != 0` in `args`')
        self.args.dim = 0
        self.network_structure = network_structure
        if 'device' in self.args:
            self.device = self.args.device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.maml = Meta(self.args, self.network_structure).to(self.device)
        self.mamls = [
            Meta(self.args, self.network_structure).to(self.device)
            for _ in range(self.args.problem_dim[1])
        ]
        self._check_integrity()
        self.dateset_to_device()
        n_obj = self.args.problem_dim[1]
        self.nets: List[List[Learner]] = [None] * n_obj
        self.fast_weights: List[List[list]] = [None] * n_obj

    def _check_integrity(self) -> None:
        """
        Check if the initial data is valid
        """
        # assert len(self.train_sets[0]) == 4
        # assert len(self.test_sets[0]) == 4
        required_attr = ['epoch', 'update_lr', 'meta_lr', 'fine_tune_lr', 'k_spt',
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
        n_obj = self.args.problem_dim[1]
        train_s, test_s = [[] for _ in range(n_obj)], [[] for _ in range(n_obj)]
        train_parted_len = len(self.train_sets) // n_obj
        train_idxs = [[i * n_obj + j for i in range(train_parted_len)] for j in range(n_obj)]
        test_parted_len = len(self.train_sets) // n_obj
        test_idxs = [[i * n_obj + j for i in range(test_parted_len)] for j in range(n_obj)]

        for arr in self.train_sets:
            [train_s[i].append(arr[train_idxs[i]]) for i in range(len(train_s))]

        for arr in self.test_sets:
            [test_s[i].append(arr[test_idxs[i]]) for i in range(len(test_s))]

        self.train_sets = [tuple([torch.from_numpy(arr).to(device) for arr in temp_train_set]) for temp_train_set in train_s]
        self.test_sets = [tuple([torch.from_numpy(arr).to(device) for arr in temp_test_set]) for temp_test_set in test_s]

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
        loss_arrs = []
        for i, which_meta in enumerate(self.mamls):
            print_loss = explicit if isinstance(explicit, bool) else explicit > 0
            print_period = 1 if isinstance(explicit, bool) else explicit

            loss_arr = []
            for epoch in range(self.args.epoch):
                if 'sgd_epoch' in self.args:
                    sgd_select_n = self.args.sgd_select_n
                    for sgd_epoch in range(self.args.sgd_epoch):
                        train = np.random.choice(np.arange(self.train_sets[i][0].shape[0], dtype=int), sgd_select_n)
                        train = [data[train] for data in self.train_sets[i]]
                        loss_arr.append(which_meta(*train))
                else:
                    loss_arr.append(which_meta(*self.train_sets[i]))
                if print_loss and (epoch % print_period == 0 or epoch == self.args.epoch - 1):
                    print(f'Epoch {epoch:4d}: {loss_arr[-1]:.4f}')
            loss_arrs.append(loss_arr)
        return loss_arrs

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
            loss = []
            for i, which_maml in enumerate(self.mamls):
                loss_, res, _nets, _fast_weights = self.maml.fine_tuning(*self.test_sets[i],
                                                                         return_single_lose=return_single_loss)
                self.nets[i] = _nets
                self.fast_weights[i] = _fast_weights
                loss.append(loss_)

        else:
            raise NotImplementedError('`pretrain=True` is not supported here')
        return loss

    def test_continue(self, x: np.ndarray, y: np.ndarray, return_single_loss: bool = True,
                      use_test_set: bool = True) -> List[List[float] | float]:
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
        if self.nets[0] is None or self.fast_weights[0] is None:
            raise ValueError('No previous fine-tuned model found, please use `test` instead')

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        x, y = x.to(self.device), y.to(self.device)
        loss = []
        for i, which_maml in enumerate(self.mamls):
            if use_test_set:
                spt_x, spt_y, qry_x, qry_y = self.test_sets[i]
            else:
                spt_x, spt_y = (None,), (None,)
                qry_x, qry_y = self.test_sets[i][2:]

            loss_, _, _nets, _fast_weights = \
                which_maml.fine_tuning_continue_multi(
                    i,
                    self.nets[i], self.fast_weights[i], x, y,
                    spt_x, spt_y, qry_x, qry_y,
                    return_single_lose=return_single_loss)
            self.nets[i] = _nets
            self.fast_weights[i] = _fast_weights
            loss.append(loss_)
        return loss

    def __call__(self, x: np.ndarray) -> List[float]:
        if self.nets[0] is None or self.fast_weights[0] is None:
            raise ValueError('No previous fine-tuned model found, please use `test` instead')
        x = torch.from_numpy(x).to(self.device)
        rets = []
        for which_net, which_fast_weights in zip(self.nets, self.fast_weights):
            ret = [net(x, which_fast_weights[i]).detach().cpu().numpy().flatten() for i, net in enumerate(which_net)]
            ret = np.array(ret).flatten()
            rets.append(ret)
        return rets

