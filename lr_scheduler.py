import numpy as np
import torch


class OneCycleScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, min_lr, beta_range, max_steps, annealing, peak_pos=0.45, end_pos=0.9, last_epoch=-1):
        assert min_lr > 0., '{} should be > 0'.format(min_lr)
        assert peak_pos < end_pos, '{} should be < {}'.format(peak_pos, end_pos)

        if annealing == 'linear':
            annealing = annealing_linear
        elif annealing == 'cosine':
            annealing = annealing_cosine
        else:
            raise AssertionError('invalid annealing {}'.format(annealing))

        self.optimizer = optimizer
        self.min_lr = min_lr
        self.beta_range = beta_range
        self.max_steps = max_steps
        self.annealing = annealing
        self.peak_pos = peak_pos
        self.end_pos = end_pos

        super().__init__(optimizer, last_epoch)

    def step(self, epoch=None):
        super().step(epoch=epoch)

        for param_group, beta in zip(self.optimizer.param_groups, self.get_beta()):
            if 'momentum' in param_group:
                param_group['momentum'] = beta
            else:
                raise AssertionError('no momentum parameter')

    def get_lr(self):
        def get_param_group_lr(base_lr):
            mid = round(self.max_steps * self.peak_pos)
            end = round(self.max_steps * self.end_pos)

            if self.last_epoch < mid:
                r = self.last_epoch / mid
                lr = self.annealing(self.min_lr, base_lr, r)
            elif self.last_epoch < end:
                r = (self.last_epoch - mid) / (end - mid)
                lr = self.annealing(base_lr, self.min_lr, r)
            else:
                r = (self.last_epoch - end) / (self.max_steps - end)
                lr = self.annealing(self.min_lr, 0, r)

            return lr

        return [get_param_group_lr(base_lr) for base_lr in self.base_lrs]

    def get_beta(self):
        def get_param_group_beta():
            mid = round(self.max_steps * self.peak_pos)
            end = round(self.max_steps * self.end_pos)

            if self.last_epoch < mid:
                r = self.last_epoch / mid
                beta = self.annealing(self.beta_range[0], self.beta_range[1], r)
            elif self.last_epoch < end:
                r = (self.last_epoch - mid) / (end - mid)
                beta = self.annealing(self.beta_range[1], self.beta_range[0], r)
            else:
                beta = self.beta_range[0]

            return beta

        return [get_param_group_beta() for _ in self.base_lrs]


class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        def get_param_group_lr(base_lr):
            if self.last_epoch < self.warmup_steps:
                r = self.last_epoch / self.warmup_steps
                lr = annealing_exponential(base_lr * 1e-3, base_lr, r)
            else:
                r = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                lr = annealing_cosine(base_lr, 0., r)

            return lr

        return [get_param_group_lr(base_lr) for base_lr in self.base_lrs]


def annealing_exponential(start, end, r):
    return start * (end / start)**r


def annealing_linear(start, end, r):
    return start + r * (end - start)


def annealing_cosine(start, end, r):
    cos_out = np.cos(np.pi * r) + 1

    return end + (start - end) / 2 * cos_out


# TODO: combine annealiing
def combine_annealing(fs, ps):
    assert len(fs) == len(ps) + 1
