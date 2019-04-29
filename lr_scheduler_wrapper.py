import numpy as np


class LRSchedulerWrapper(object):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def step(self):
        pass

    def step_epoch(self):
        pass

    def step_score(self, score):
        pass

    def get_lr(self):
        lr = []
        beta = []

        for param_group in self.scheduler.optimizer.param_groups:
            lr.append(param_group['lr'])

            if 'betas' in param_group:
                beta.append(param_group['betas'][0])
            elif 'momentum' in param_group:
                beta.append(param_group['momentum'])
            else:
                raise AssertionError('no beta parameter')

        return np.squeeze(lr), np.squeeze(beta)


class StepWrapper(LRSchedulerWrapper):
    def step(self):
        return self.scheduler.step()


class EpochWrapper(LRSchedulerWrapper):
    def step_epoch(self):
        return self.scheduler.step()


class ScoreWrapper(LRSchedulerWrapper):
    def step_score(self, score):
        return self.scheduler.step(score)
