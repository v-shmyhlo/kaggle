import numpy as np


class OneCycleScheduler(object):
    def __init__(self, optimizer, lr, beta, max_steps, annealing):
        self.optimizer = optimizer
        self.lr = lr
        self.beta = beta
        self.max_steps = max_steps
        self.annealing = annealing
        self.epoch = -1

    def step(self):
        self.epoch += 1

        lr, beta = self.get_lr()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

            if 'betas' in param_group:
                param_group['betas'] = (beta, *param_group['betas'][1:])
            elif 'momentum' in param_group:
                param_group['momentum'] = beta
            else:
                raise AssertionError('no beta parameter')

    def get_lr(self):
        mid = round(self.max_steps * 0.3)

        if self.epoch < mid:
            r = self.epoch / mid
            lr = self.annealing(self.lr[0], self.lr[1], r)
            beta = self.annealing(self.beta[0], self.beta[1], r)
        else:
            r = (self.epoch - mid) / (self.max_steps - mid)
            lr = self.annealing(self.lr[1], self.lr[0] / 1e4, r)
            beta = self.annealing(self.beta[1], self.beta[0], r)

        return lr, beta


def annealing_linear(start, end, r):
    return start + r * (end - start)


def annealing_cos(start, end, r):
    cos_out = np.cos(np.pi * r) + 1

    return end + (start - end) / 2 * cos_out
