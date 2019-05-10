import numpy as np


# TODO: refactor to use optimizers LR (check torch code)


class LRScheduler(object):
    def step(self):
        raise NotImplementedError

    def get_lr(self):
        raise NotImplementedError


class OneCycleScheduler(LRScheduler):
    def __init__(self, optimizer, lr, beta, max_steps, annealing, peak_pos=0.3, end_pos=0.9):
        assert peak_pos < end_pos, '{} should be less than {}'.format(peak_pos, end_pos)

        if annealing == 'linear':
            annealing = annealing_linear
        elif annealing == 'cosine':
            annealing = annealing_cosine
        else:
            raise AssertionError('invalid annealing {}'.format(annealing))

        self.optimizer = optimizer
        self.lr = lr
        self.beta = beta
        self.max_steps = max_steps
        self.annealing = annealing
        self.peak_pos = peak_pos
        self.end_pos = end_pos
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1

        lr = self.get_lr()
        beta = self.get_beta()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

            if 'betas' in param_group:
                param_group['betas'] = (beta, *param_group['betas'][1:])
            elif 'momentum' in param_group:
                param_group['momentum'] = beta
            else:
                raise AssertionError('no beta parameter')

    def get_lr(self):
        mid = round(self.max_steps * self.peak_pos)
        end = round(self.max_steps * self.end_pos)

        if self.last_epoch < mid:
            r = self.last_epoch / mid
            lr = self.annealing(self.lr[0], self.lr[1], r)
        elif self.last_epoch < end:
            r = (self.last_epoch - mid) / (end - mid)
            lr = self.annealing(self.lr[1], self.lr[0], r)
        else:
            r = (self.last_epoch - end) / (self.max_steps - end)
            lr = self.annealing(self.lr[0], self.lr[0] * 1e-4, r)

        return lr

    def get_beta(self):
        mid = round(self.max_steps * self.peak_pos)
        end = round(self.max_steps * self.end_pos)

        if self.last_epoch < mid:
            r = self.last_epoch / mid
            beta = self.annealing(self.beta[0], self.beta[1], r)
        elif self.last_epoch < end:
            r = (self.last_epoch - mid) / (end - mid)
            beta = self.annealing(self.beta[1], self.beta[0], r)
        else:
            beta = self.beta[0]

        return beta


class LinearScheduler(LRScheduler):
    def __init__(self, optimizer, delta):
        self.optimizer = optimizer
        self.delta = delta
        self.lr_initial = np.squeeze([param_group['lr'] for param_group in optimizer.param_groups])
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1

        lr = self.get_lr()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        return self.lr_initial + self.delta * self.last_epoch


def annealing_linear(start, end, r):
    return start + r * (end - start)


def annealing_cosine(start, end, r):
    cos_out = np.cos(np.pi * r) + 1

    return end + (start - end) / 2 * cos_out
