import numpy as np
from pymoo.core.problem import Problem
from pymoo.problems.many.dtlz import get_ref_dirs

from maml_mod import MamlWrapper


class DTLZb(Problem):
    def __init__(self, n_var, n_obj, delta1, delta2, k=None):
        self.delta1 = delta1
        self.delta2 = delta2

        if n_var:
            self.k = n_var - n_obj + 1
        elif k:
            self.k = k
            n_var = k + n_obj - 1
        else:
            raise Exception("Either provide number of variables or k!")

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=0, xu=1, type_var=np.double)

    def g1(self, X_M):
        return (100 + self.delta1) * (
                self.k + self.delta2 + np.sum(np.square(X_M - 0.5) - np.cos(20 * np.pi * (X_M - 0.5)), axis=1))

    def g2(self, X_M):
        return np.sum(np.square(X_M - 0.5), axis=1)

    def obj_func(self, X_, g, alpha=1):
        f = []

        for i in range(0, self.n_obj):
            _f = (1 + g)
            _f *= np.prod(np.cos(np.power(X_[:, :X_.shape[1] - i], alpha) * np.pi / 2.0), axis=1)
            if i > 0:
                _f *= np.sin(np.power(X_[:, X_.shape[1] - i], alpha) * np.pi / 2.0)

            f.append(_f)

        f = np.column_stack(f)
        return f


class DTLZ1b(DTLZb):
    def __init__(self, n_var=7, n_obj=3, delta1=0, delta2=0, **kwargs):
        self.delta1 = delta1
        self.delta2 = delta2
        super().__init__(n_var, n_obj, delta1, delta2, **kwargs)

    def _calc_pareto_front(self, ref_dirs=None):
        if ref_dirs is None:
            ref_dirs = get_ref_dirs(self.n_obj)
        coefficient = max(0.5, 0.5 * (100 + self.delta1) * self.delta2)
        return coefficient * ref_dirs

    def obj_func(self, X_, g):
        f = []

        for i in range(0, self.n_obj):
            _f = 0.5 * (1 + g)
            _f *= np.prod(X_[:, :X_.shape[1] - i], axis=1)
            if i > 0:
                _f *= 1 - X_[:, X_.shape[1] - i]
            f.append(_f)

        return np.column_stack(f)

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)
        out["F"] = self.obj_func(X_, g)


class DTLZbProblem(Problem):
    def __init__(self, sol: MamlWrapper):
        self.sol = sol
        super().__init__(n_var=8,
                         n_obj=3,
                         #  n_constr=2,
                         xl=np.array([0] * 8, np.float32),
                         xu=np.array([1] * 8, np.float32),
                         )

    def _evaluate(self, x, out, *args, **kwargs):
        x = x.astype(np.float32)
        f = []
        for xi in x:
            fi = self.sol(xi)
            f.append(fi)
        out["F"] = np.array(f)
