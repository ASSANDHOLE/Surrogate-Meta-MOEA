import numpy as np
from pymoo.core.problem import Problem

from maml_mod import MamlWrapper


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
