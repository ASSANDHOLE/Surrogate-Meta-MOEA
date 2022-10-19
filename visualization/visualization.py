from typing import List, Literal

import numpy as np
import matplotlib.pyplot as plt


def visualize_loss(maml_loss: List[float], baseline_loss: List[float],
                   title: str = 'Loss', alignment: Literal['max', 'maml', 'baseline'] = 'max') -> None:
    """
    Visualize the loss of the MAML and Baseline

    Parameters
    ----------
    maml_loss : List[float]
        The loss of the MAML
    baseline_loss : List[float]
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
    plt.plot(maml_loss, label='MAML')
    plt.plot(baseline_loss, label='Baseline')
    plt.title(title)
    plt.xlabel('Gradient Steps')
    plt.ylabel('Loss')
    plt.legend()
    if alignment == 'max':
        alignment = max(len(maml_loss), len(baseline_loss))
    elif alignment == 'maml':
        alignment = len(maml_loss)
    elif alignment == 'baseline':
        alignment = len(baseline_loss)

    plt.xlim(0, alignment)
    plt.show()
