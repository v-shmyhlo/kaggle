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
        for param_group in self.scheduler.optimizer.param_groups:
            lr.append(param_group['lr'])

        return np.squeeze(lr)


class StepWrapper(LRSchedulerWrapper):
    def step(self):
        return self.scheduler.step()


class EpochWrapper(LRSchedulerWrapper):
    def step_epoch(self):
        return self.scheduler.step()


class ScoreWrapper(LRSchedulerWrapper):
    def step_score(self, score):
        return self.scheduler.step(score)
