from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from .learner import Learner


class Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        imgc = 'imgc' in args and args.imgc or None
        imgsz = 'imgsz' in args and args.imgsz or None
        self.net = Learner(config, imgc=imgc, imgsz=imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm / counter

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        loss_func = F.mse_loss
        task_num, setsz, *data_size = x_spt.size()
        querysz = x_qry.size(1)

        if len(y_qry.shape) == 2:
            y_qry = y_qry.unsqueeze(2)

        if len(y_spt.shape) == 2:
            y_spt = y_spt.unsqueeze(2)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i

        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = loss_func(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = loss_func(logits_q, y_qry[i])
                losses_q[0] += loss_q

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = loss_func(logits_q, y_qry[i])
                losses_q[1] += loss_q

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = loss_func(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = loss_func(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()

        return loss_q.item()

    def _fine_tuning(self, x_spt, y_spt, x_qry, y_qry, net=None, return_single_lose=True):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :param return_single_lose: if True, return a single loss, else return a list of losses
        :return:
        """
        loss_func = F.mse_loss

        assert len(x_spt.shape) >= 2

        query_size = x_qry.size(0)
        if query_size != 0:
            if len(y_qry.shape) == 1:
                y_qry = y_qry.unsqueeze(1)

            if y_qry.shape[-1] != 1:
                y_qry = y_qry.unsqueeze(-1)

        if len(y_spt.shape) == 1:
            y_spt = y_spt.unsqueeze(1)

        if y_spt.shape[-1] != 1:
            y_spt = y_spt.unsqueeze(-1)

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(net) if net is not None else deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = loss_func(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))
        losses = []

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = loss_func(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            if query_size != 0:
                logits_q = net(x_qry, fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                losses.append(loss_func(logits_q, y_qry).item())

        ret_loss = losses[-1] if return_single_lose else losses

        return None if query_size == 0 else ret_loss, None if query_size == 0 else logits_q, net, fast_weights

    def fine_tuning(self, x_spt, y_spt, x_qry, y_qry, net=None, return_single_lose=True):
        n_net = x_spt.size(0)
        ret = [[], [], [], []]
        for i in range(n_net):
            x, y = x_spt[i].reshape(1, *x_spt.size()[1:]), y_spt[i].reshape(1, *y_spt.size()[1:])
            x_q, y_q = x_qry[i].reshape(1, *x_qry.size()[1:]), y_qry[i].reshape(1, *y_qry.size()[1:])
            t_loss, t_logits, t_net, t_weights = self._fine_tuning(x, y, x_q, y_q, net, return_single_lose)
            ret[0].append(t_loss)
            ret[1].append(t_logits)
            ret[2].append(t_net)
            ret[3].append(t_weights)
        return tuple(ret)

    def _fine_tuning_continue(self, net, fast_weights, x, y, x_spt, y_spt, x_qry, y_qry, return_single_lose=True):
        loss_func = F.mse_loss
        query_size = 0 if not hasattr(x_qry, 'size') else x_qry.size(0)

        if x_spt is not None:
            x_spt = torch.cat([x_spt, x], 0)
            y_spt = torch.cat([y_spt, y], 0)
        else:
            x_spt = x
            y_spt = y

        if query_size != 0:
            if len(y_qry.shape) == 1:
                y_qry = y_qry.unsqueeze(1)

            if y_qry.shape[-1] != 1:
                y_qry = y_qry.unsqueeze(-1)

        if len(y_spt.shape) == 1:
            y_spt = y_spt.unsqueeze(1)

        if y_spt.shape[-1] != 1:
            y_spt = y_spt.unsqueeze(-1)

        losses = []

        for k in range(0, self.update_step_test):
            logits = net(x_spt, fast_weights, bn_training=True)
            y_spt = y_spt.reshape(logits.shape)
            loss = loss_func(logits, y_spt)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            if query_size != 0:
                logits_q = net(x_qry, fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                losses.append(loss_func(logits_q, y_qry).item())

        if query_size != 0:
            ret_loss = losses[-1] if return_single_lose else losses
        else:
            ret_loss = None

        return ret_loss, None if query_size == 0 else logits_q, net, fast_weights

    def fine_tuning_continue(self, nets, fast_weights, x, y, x_spt, y_spt, x_qry, y_qry, return_single_lose=True):
        assert x.shape[0] == len(nets)
        n_net = len(nets)
        ret = [[], [], [], []]
        for i in range(n_net):
            if x_spt is not None:
                x_s, y_s = x_spt[i].reshape(1, *x.size()[1:]), y_spt[i].reshape(1, *y.size()[1:])
            else:
                x_s, y_s = None, None
            if x_qry is not None:
                x_q, y_q = x_qry[i].reshape(1, *x_qry.size()[1:]), y_qry[i].reshape(1, *y_qry.size()[1:])
            else:
                x_q, y_q = None, None
            t_loss, t_logits, t_net, t_weights = self._fine_tuning_continue(
                nets[i], fast_weights[i], x[i], y[i], x_s, y_s, x_q, y_q, return_single_lose
            )
            ret[0].append(t_loss)
            ret[1].append(t_logits)
            ret[2].append(t_net)
            ret[3].append(t_weights)
        return tuple(ret)

    def pretrain_fine_tuning(self, train_x, train_y, x_spt, y_spt, x_qry, y_qry, return_single_lose=True):

        loss_func = F.mse_loss

        assert len(x_spt.shape) >= 2

        querysz = x_qry.size(0)

        if len(y_qry.shape) == 1:
            y_qry = y_qry.unsqueeze(1)

        if len(y_spt.shape) == 1:
            y_spt = y_spt.unsqueeze(1)

        if y_qry.shape[-1] != 1:
            y_qry = y_qry.unsqueeze(-1)

        if y_spt.shape[-1] != 1:
            y_spt = y_spt.unsqueeze(-1)

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # pretrain
        # 1. run the i-th task and compute loss for k=0
        logits = net(train_x)
        loss = loss_func(logits, train_y)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(train_x, fast_weights, bn_training=True)
            loss = loss_func(logits, train_y)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

        # 1. run the i-th task and compute loss for k=0
        losses, _, _, _ = self.fine_tuning(x_spt, y_spt, x_qry, y_qry, net, return_single_lose)
        return losses
