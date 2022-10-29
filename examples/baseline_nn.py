from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class BaselineNn(nn.Module):
    def __init__(self, n_dim: Tuple[int, int]):
        super(BaselineNn, self).__init__()
        self.n_args_in, self.n_args_out = n_dim
        self.layers = nn.Sequential(
            nn.Linear(self.n_args_in, 2 * self.n_args_in),
            nn.ReLU(),
            nn.Linear(2 * self.n_args_in, 4 * self.n_args_in),
            nn.ReLU(),
            nn.Linear(4 * self.n_args_in, 4 * self.n_args_in),
            nn.ReLU(),
            nn.Linear(4 * self.n_args_in, 2 * self.n_args_in),
            nn.ReLU(),
            nn.Linear(2 * self.n_args_in, self.n_args_out),
        )

    def forward(self, x, return_tensor=False):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.args.device)
        self.to(self.args.device)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        ret = self.layers(x)
        return ret if return_tensor else ret.detach().cpu().numpy()


class Wrapper:
    def __init__(self, shape):
        self.shape = shape
        self.models = [BaselineNn((shape[0], 1)) for _ in range(shape[1])]

    def __call__(self, x, *args, **kwargs):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.args.device)
        for m in self.models:
            m.__dict__['args'] = self.args
        ys = []
        for i in range(self.shape[1]):
            ys.append(self.models[i](x).flatten()[0])
        ret = np.array(ys)
        if 'return_tensor' in kwargs and kwargs['return_tensor']:
            return torch.from_numpy(ret).float().to(self.args.device)
        return ret


__model = None


def train(x: torch.Tensor, y: torch.Tensor, shape: Tuple[int, int],
          init: bool = False, lr: float = 0.001, n_epochs: int = 1000):
    global __model
    if init:
        __model = Wrapper(shape)
        return __model
    for i, s in enumerate(__model.models):
        optimizer = torch.optim.Adam(s.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        y_tensor = torch.from_numpy(y[i]).float().to(s.args.device)
        loss_list = []
        for _ in range(n_epochs):
            optimizer.zero_grad()
            y_pred = s(x, return_tensor=True)
            loss = loss_fn(y_pred, y_tensor)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

        print(f'Loss: {loss_list[-1]}, avg loss: {np.mean(loss_list)}+-{np.std(loss_list)}')

    return __model
