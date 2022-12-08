from typing import Tuple, List

import numpy as np
import torch

from maml_mod import MrMamlAct as Meta
from maml_mod import Learner
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
            if 'sgd_epoch' in self.args:
                sgd_select_n = self.args.sgd_select_n
                for sgd_epoch in range(self.args.sgd_epoch):
                    train = np.random.choice(np.arange(self.train_set[0].shape[0], dtype=int), sgd_select_n)
                    train = [data[train] for data in self.train_set]
                    loss_arr.append(self.maml(*train))
            else:
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
        if self.nets is None or self.fast_weights is None:
            raise ValueError('No previous fine-tuned model found, please use `test` instead')

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        x, y = x.to(self.device), y.to(self.device)

        if use_test_set:
            spt_x, spt_y, qry_x, qry_y = self.test_set
        else:
            spt_x, spt_y = (None,), (None,)
            qry_x, qry_y = self.test_set[2:]

        loss, _, _nets, _fast_weights = self.maml.fine_tuning_continue(self.nets, self.fast_weights, x, y,
                                                                       spt_x, spt_y, qry_x, qry_y,
                                                                       return_single_lose=return_single_loss)
        self.nets = _nets
        self.fast_weights = _fast_weights
        return loss

    def __call__(self, x: np.ndarray) -> List[float]:
        if self.nets is None or self.fast_weights is None:
            raise ValueError('No previous fine-tuned model found, please use `test` instead')
        x = torch.from_numpy(x).to(self.device)
        ret = [net(x, self.fast_weights[i]).detach().cpu().numpy().flatten() for i, net in enumerate(self.nets)]
        ret = np.array(ret).flatten()
        return ret

