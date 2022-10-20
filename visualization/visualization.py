from __future__ import annotations

from typing import List, Literal, Any

import numpy as np
import matplotlib.pyplot as plt


def visualize_loss(maml_loss: List[Any] | np.ndarray, baseline_loss: List[Any] | np.ndarray,
                   title: str = 'Loss', alignment: Literal['max', 'maml', 'baseline'] = 'max') -> None:
    """
    Visualize the loss of the MAML and Baseline

    Parameters
    ----------
    maml_loss : List[float] | np.ndarray
        The loss of the MAML
    baseline_loss : List[float] | np.ndarray
        The loss of the Baseline
    title : str
        The title of the plot
    alignment : Literal['max', 'maml', 'baseline']
        The alignment of the plot, default to 'max'
        if 'max', the plot will be aligned to the maximum length of the two losses;
        if 'maml', the plot will be aligned to the length of the MAML loss;
        if 'baseline', the plot will be aligned to the length of the Baseline loss;
    """
    plt.figure()

    def plot_loss(loss: List[Any] | np.ndarray, label: str) -> int:
        if hasattr(loss[0], '__len__'):
            for i in range(len(loss)):
                plt.plot(loss[i], label=f'{label}-{i + 1}')
            return len(loss[0])
        else:
            plt.plot(maml_loss, label='MAML')
            return len(loss)

    len_maml = plot_loss(maml_loss, 'MAML')
    len_baseline = plot_loss(baseline_loss, 'Baseline')

    plt.title(title)
    plt.xlabel('Gradient Steps')
    plt.ylabel('Loss')
    plt.legend()
    if alignment == 'max':
        alignment = max(len_maml, len_baseline)
    elif alignment == 'maml':
        alignment = len_maml
    elif alignment == 'baseline':
        alignment = len_baseline

    plt.xlim(0, alignment)
    plt.show()
