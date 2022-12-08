import sys
from typing import Tuple

from IPython.core import ultratb


class NamedDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.__dict__ = self

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            return AttributeError


class _IPythonExceptionHook:
    instance = None

    def __call__(self, *args, **kwargs):
        if self.instance is None:
            self.instance = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)
        self.instance(*args, **kwargs)


def set_ipython_exception_hook():
    sys.excepthook = _IPythonExceptionHook()


def test(_n, _m):
    import numpy as np
    tot = _n
    each = _m
    tot_arr = np.arange(tot, dtype=np.int32)
    sel = np.zeros_like(tot_arr, dtype=bool)
    i = 0
    while np.sum(sel) < tot:
        i += 1
        sel[np.random.choice(tot_arr, each)] = True
    # print(i)
    return i


def calculate_confidence_k(task_num: int,
                           select_n_from_task: int,
                           k_range: Tuple[int, int] = (50, 300),
                           minimum_confidence: float = 0.95) -> int:
    """
    Calculate the minimum k that can guarantee the confidence of selecting all tasks\

    Parameters
    ----------
    task_num : int
        The number of tasks to select
    select_n_from_task : int
        The number of tasks to be sampled from all tasks
    k_range : Tuple[int, int]
        The range of k to search, k is n_run
    minimum_confidence : float
        The minimum confidence range(0, 1) to guarantee

    Returns
    -------
    int:
        The minimum k that can guarantee the confidence of selecting all tasks
        If k is at the end of the range, the confidence is not guaranteed
    """
    m, n, ks = select_n_from_task, task_num, list(range(*k_range))
    for k in ks:
        p = ((n ** k - (n - m) ** k) / n ** k) ** n
        if p >= minimum_confidence:
            return k
    return ks[-1]


def draw_curve(n, m):
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    k = list(range(10, 200))
    y = [((n ** ki - (n - m) ** ki) / n ** ki) ** n for ki in k]
    arr = [test(n, m) for _ in range(2000)]
    plt.hist(arr, bins=50, density=True)
    div = np.diff(y) / np.diff(k)
    div = np.concatenate(([div[0]], div))
    # plt.plot(k, y)
    plt.plot(k, div)
    plt.show()


if __name__ == '__main__':
    _n, _m = 2000, 100
    # draw_curve(_n, _m)
    v = calculate_confidence_k(_n, _m)
    print(v)
