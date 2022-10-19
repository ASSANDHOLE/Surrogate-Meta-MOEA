# Unified MAML for this project
from __future__ import annotations

from typing import Tuple, List

import numpy as np
import torch

from maml_mod import Meta
from visualization import visualize_loss

from utils import NamedDict
from example import get_args, get_network_structure, get_dataset
from example_sinewave import get_args_maml_regression, get_network_structure_maml_regression, get_dataset_sinewave


class Sol:
    def __init__(self,
                 dataset: Tuple[
                     Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                     Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
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
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
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

    def test(self, return_single_loss: bool = True, pretrain: bool = False) -> Tuple[np.ndarray, float | List[float]]:
        """
        Test the model

        Parameters
        ----------
        return_single_loss : bool
            If True, the loss of each gradient update will be returned

        Returns
        -------
        Tuple[np.ndarray, float]:
            The first element is the output of the model
            The second element is the loss of the model
        """
        if not pretrain:
            loss, res = self.maml.finetunning(*self.test_set, return_single_lose=return_single_loss)
        else:
            train_x = self.train_set[0].reshape((1, -1, self.train_set[0].shape[-1]))
            train_y = self.train_set[1].reshape((1, -1, 1))
            loss, res = self.maml.pretrain_finetunning(train_x, train_y, *self.test_set,
                                                       return_single_lose=return_single_loss)
        return res, loss


def main():
    # see Sol.__init__ for more information
    args = get_args()
    network_structure = get_network_structure(args)
    dataset = get_dataset(args, normalize_targets=True)
    sol = Sol(dataset, args, network_structure)
    train_loss = sol.train(explicit=5)
    test_res, test_loss = sol.test()
    print(f'Test loss: {test_loss:.4f}')

    args.test_update_step = 30
    sol = Sol(dataset, args, network_structure)
    random_res, random_loss = sol.test()
    print(f'Random loss: {random_loss:.4f}')


def main_sinewave():
    args = get_args_maml_regression()
    network_structure = get_network_structure_maml_regression()
    dataset = get_dataset_sinewave(args, normalize_targets=True)
    sol = Sol(dataset, args, network_structure)
    train_loss = sol.train(explicit=5)
    test_res, test_loss = sol.test(return_single_loss=False)
    print(f'Test loss: {test_loss[-1]:.4f}')

    args.update_step_test = int(1.5 * args.update_step_test)
    sol = Sol(dataset, args, network_structure)
    random_res, random_loss = sol.test(return_single_loss=False, pretrain=True)
    print(f'Random loss: {random_loss[-1]:.4f}')
    visualize_loss(test_loss, random_loss)


if __name__ == '__main__':
    # main()
    main_sinewave()
