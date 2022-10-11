# Unified MAML for this project

from typing import Tuple, List

import numpy as np
import torch

from maml_mod import Meta

from utils import NamedDict


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
        required_attr = ['epoch', 'update_lr', 'meta_lr', 'n_way', 'k_spt',
                         'k_qry', 'task_num', 'update_step', 'update_step_test']
        assert all([hasattr(self.args, attr) for attr in required_attr])

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

    def test(self) -> Tuple[np.ndarray, float]:
        """
        Test the model

        Returns
        -------
        Tuple[np.ndarray, float]:
            The first element is the output of the model
            The second element is the loss of the model
        """
        loss, res = self.maml.finetunning(*self.test_set)
        return res, loss


def main():
    # fixme: change the following declaration to your own
    # see Sol.__init__ for more information
    dataset = None
    args = None
    network_structure = None
    sol = Sol(dataset, args, network_structure)
    train_loss = sol.train(explicit=10)
    test_res, test_loss = sol.test()
    print(f'Test loss: {test_loss:.4f}')


if __name__ == '__main__':
    main()
