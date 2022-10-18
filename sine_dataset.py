import math
import random
import numpy as np
from typing import Tuple, List, Callable


def get_sine_wave_sampling(amp: float = 1, phase: float = 0, num_points: int = 100,
                           domain: Tuple[float, float] = (0, 1)) ->\
        Tuple[List[Tuple[float, float]], Callable[[float], float]]:

    def sine_wave_func(x: float) -> float:
        return math.sin(x + phase) * amp

    # uniform sampling
    interval = (domain[1] - domain[0]) / num_points
    X = [domain[0] + i * interval + interval / 2 for i in range(num_points)]
    dataset = [(x, sine_wave_func(x)) for x in X]

    return dataset, sine_wave_func


def create_dataset_sinewave(problem_dim: Tuple[int, int], train_test: Tuple[int, int], spt_qry: Tuple[int, int],
                            amp: Tuple[float, float] = (0.1, 5.0), phase: Tuple[float, float] = (0.1, math.pi),
                            domain: Tuple[float, float] = (-5.0, 5.0)) -> \
        Tuple[
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ]:
    if problem_dim[0] != 1:
        print('The objective function of the dataset only support 1 variable. \n'
              'Current number of variables: %d', problem_dim[0])
        exit(-1)

    # generate train set
    num_points = spt_qry[0] + spt_qry[1]

    def create_dataset_inner(n_problems: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        set_spt_x, set_spt_y, set_qry_x, set_qry_y = [], [], [], []
        for objective in range(problem_dim[1] * n_problems):
            _amp = amp[0] + random.random() * (amp[1]-amp[0])  # [0.1, 5.0)
            _phase = phase[0] + random.random() * (phase[1]-phase[0])  # [0, 2pi)
            dataset, _ = get_sine_wave_sampling(_amp, _phase, num_points, domain)
            random.shuffle(dataset)
            obj_spt, obj_qry = dataset[:spt_qry[0]], dataset[spt_qry[0]:]
            obj_spt_x, obj_spt_y = zip(*obj_spt)
            obj_qry_x, obj_qry_y = zip(*obj_qry)
            set_spt_x.append(obj_spt_x)
            set_spt_y.append(obj_spt_y)
            set_qry_x.append(obj_qry_x)
            set_qry_y.append(obj_qry_y)
        set_spt_x = np.array(set_spt_x).astype(np.float32)[:, :, np.newaxis]
        set_spt_y = np.array(set_spt_y).astype(np.float32)
        set_qry_x = np.array(set_qry_x).astype(np.float32)[:, :, np.newaxis]
        set_qry_y = np.array(set_qry_y).astype(np.float32)
        return set_spt_x, set_spt_y, set_qry_x, set_qry_y

    train_spt_x, train_spt_y, train_qry_x, train_qry_y = create_dataset_inner(train_test[0])
    test_spt_x, test_spt_y, test_qry_x, test_qry_y = create_dataset_inner(train_test[1])
    return (train_spt_x, train_spt_y, train_qry_x, train_qry_y), (test_spt_x, test_spt_y, test_qry_x, test_qry_y)


def test():
    train_set, test_set = create_dataset_sinewave(problem_dim=(1, 3), train_test=(4, 2), spt_qry=(5, 20))  # (12, 5, 1)
    print(train_set[0].shape, train_set[1].shape, train_set[2].shape, train_set[3].shape)
    print(test_set[0].shape, test_set[1].shape, test_set[2].shape, test_set[3].shape)
    print(train_set[0])


if __name__ == '__main__':
    test()
