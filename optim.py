import math
from collections import defaultdict

import torch


class DummySwitchable(torch.optim.Optimizer):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.defaults = self.optimizer.defaults
        self.param_groups = self.optimizer.param_groups
        self.training = False

    def step(self, closure=None):
        assert self.training

        loss = self.optimizer.step(closure)

        return loss

    def train(self):
        assert not self.training
        self.training = True

    def eval(self):
        assert self.training
        self.training = False


class EWA(torch.optim.Optimizer):
    def __init__(self, optimizer, momentum, num_steps):
        self.optimizer = optimizer
        self.num_steps = num_steps
        self.defaults = self.optimizer.defaults
        self.param_groups = self.optimizer.param_groups
        self.training = False

        fail

        for group in self.param_groups:
            group['ewa_step_counter'] = 0
            group['ewa_momentum'] = momentum
            group['ewa_params'] = []
            group['ewa_saved_params'] = []
            for p in group['params']:
                ewa_p = torch.empty_like(p.data)
                ewa_p.copy_(p.data)
                group['ewa_params'].append(ewa_p)

                ewa_saved_p = torch.empty_like(p.data)
                ewa_saved_p.copy_(p.data)
                group['ewa_saved_params'].append(ewa_saved_p)

    def update_ewa_group(self, group):
        assert len(group['params']) == len(group['ewa_params']) == len(group['ewa_saved_params'])

        for p, ewa_p in zip(group['params'], group['ewa_params']):
            mom = group['ewa_momentum']
            ewa_p.mul_(mom).add_(1 - mom, p.data)

    def step(self, closure=None):
        assert self.training

        loss = self.optimizer.step(closure)

        for group in self.param_groups:
            group['ewa_step_counter'] += 1
            step_counter = group['ewa_step_counter']

            if step_counter % self.num_steps == 0:
                self.update_ewa_group(group)

        return loss

    def train(self):
        assert not self.training
        self.training = True

        for group in self.param_groups:
            for p, ewa_p, ewa_saved_p in zip(group['params'], group['ewa_params'], group['ewa_saved_params']):
                p.data.copy_(ewa_saved_p)

    def eval(self):
        assert self.training
        self.training = False

        for group in self.param_groups:
            for p, ewa_p, ewa_saved_p in zip(group['params'], group['ewa_params'], group['ewa_saved_params']):
                ewa_saved_p.copy_(p.data)
                p.data.copy_(ewa_p)


# TODO: https://github.com/alphadl/lookahead.pytorch/blob/master/lookahead.py
class LA(torch.optim.Optimizer):
    def __init__(self, optimizer, lr, num_steps):
        self.num_steps = num_steps
        self.optimizer = optimizer
        self.defaults = self.optimizer.defaults
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.opt_state = self.optimizer.state

        for group in self.param_groups:
            group['la_step_counter'] = 0
            group['la_lr'] = lr

    def update_la_group(self, group):
        assert len(group['params']) == len(group['la_params'])

        for p in zip(group['params'], group['la_params']):
            param_state = self.state[p]

            if 'la_params' not in param_state:
                param_state['la_params'] = torch.empty_like(p.data)
                param_state['la_params'].copy_(p.data)

            la_p = param_state['la_params']
            la_p.add_(group['la_lr'], p.data - la_p)
            p.data.copy_(la_p)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)

        for group in self.param_groups:
            group['la_step_counter'] += 1
            step_counter = group['la_step_counter']

            if step_counter % self.num_steps == 0:
                self.update_la_group(group)

        return loss


class AdamW(torch.optim.Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1**state['step']
                bias_correction2 = 1 - beta2**state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)
                # TODO: check
                if group['weight_decay'] != 0:
                    mul = torch.tensor(group['weight_decay']).to(p.data.device)
                    p.data.addcmul_(-step_size, p.data, mul)

        return loss
