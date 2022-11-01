from __future__ import annotations

import os
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


def file_exist_or_alternate(file_path: str, save_alternative=True) -> str | None:
    """
    Check if the file exists, if it does and `save_alternative`, return the next available file name.

    Parameters
    ----------
    file_path : str
        The file path
    save_alternative : bool
        Whether to return the next available file name if the file exists

    Returns
    -------
    str | None
        The file path if the file does not exist;
        the next available file name if the file exists and `save_alternative` is True.
        None if the file exists and `save_alternative` is False.
    """
    if os.path.exists(file_path):
        if save_alternative:
            file_dir, file_name = os.path.split(file_path)
            file_name, suffix = os.path.splitext(file_name)
            i = 1
            while os.path.exists(os.path.join(file_dir, f'{file_name}_{i}{suffix}')):
                i += 1
            return os.path.join(file_dir, f'{file_name}_{i}{suffix}')
        else:
            return None
    else:
        return file_path


def show_and_save(show: bool, save_path: str | None, save_alternative: bool) -> str | None:
    if show:
        plt.show()
    if save_path is not None:
        save_path = file_exist_or_alternate(save_path, save_alternative)
        if save_path is not None:
            plt.savefig(save_path)

    return save_path


def visualize_pf(pf, label, color, scale=None, pf_true=None, show=True, save_path=None, save_alternative=True):
    # if `save_path` is None, the figure will not be saved.

    # if `save_alternative_name` is True, the figure will be saved with the next available name
    #   in the format of `save_path_dir/save_path_name_{i}.suffix` where `i` is the next available number.
    plt.figure(figsize=(8, 6))
    ax = plt.axes(projection='3d')
    ax.scatter3D(pf[:, 0], pf[:, 1], pf[:, 2], color=color, label=label)
    if pf_true is not None:
        ax.scatter3D(pf_true[:, 0], pf_true[:, 1], pf_true[:, 2], color='y', label='True Parato Front')
    if scale is not None:
        ax.set_xlim(0, scale[0])
        ax.set_ylim(0, scale[1])
        ax.set_zlim(0, scale[2])
    ax.legend(loc='best')
    ax.set(xlabel="F_1", ylabel="F_2", zlabel="F_3")
    res = show_and_save(show, save_path, save_alternative)
    if res is not None:
        print(f'Figure saved to {res}')
    if save_path is not None and res is None:
        print(f'Figure already exists at {save_path}')


def visualize_igd(func_evals, igds, colors, labels, show=True, save_path=None, save_alternative=True):
    plt.figure(figsize=(8, 6))
    for i in range(len(igds)):
        plt.plot(func_evals[i], igds[i], color=colors[i], lw=0.7, label=labels[i])
        plt.scatter(func_evals[i], igds[i], facecolor="none", edgecolor=colors[i], marker="p")
    plt.axhline(10**-1, color="red", label="10^-1", linestyle="--")
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("IGD")
    plt.yscale("log")
    plt.legend()
    res = show_and_save(show, save_path, save_alternative)
    if res is not None:
        print(f'Figure saved to {res}')
    if save_path is not None and res is None:
        print(f'Figure already exists at {save_path}')
