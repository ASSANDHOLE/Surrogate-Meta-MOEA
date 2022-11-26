import sys

from IPython.core import ultratb


class NamedDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

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


def test():
    import numpy as np
    tot = 900
    each = 50
    tot_arr = np.arange(tot, dtype=np.int32)
    sel = np.zeros_like(tot_arr, dtype=bool)
    i = 0
    while np.sum(sel) < tot:
        i += 1
        sel[np.random.choice(tot_arr, each)] = True
    # print(i)
    return i


def draw_curve():
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    m = 100
    n = 900
    k = list(range(10, 200))
    y = [((n**ki - (n-m)**ki) / n**ki) ** n for ki in k]
    arr = [test() for _ in range(2000)]
    plt.hist(arr, bins=50, density=True)
    div = np.diff(y) / np.diff(k)
    div = np.concatenate(([div[0]], div))
    # plt.plot(k, y)
    plt.plot(k, div)
    plt.show()


if __name__ == '__main__':
    draw_curve()

