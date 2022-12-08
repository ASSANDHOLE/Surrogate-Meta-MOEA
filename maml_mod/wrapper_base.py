from typing import Tuple, List

from abc import ABC, abstractmethod

import numpy as np

from utils import NamedDict


class MamlWrapperAbc(ABC):
    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def __call__(self, x: np.ndarray) -> List[float]:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError
