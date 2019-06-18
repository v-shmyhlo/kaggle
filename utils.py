import random
import warnings
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import scipy.signal
import os
import numpy as np
import torch


class EWA(object):
    def __init__(self, beta=0.9):
        self.beta = beta
        self.step = 0
        self.average = 0

    def update(self, value):
        self.step += 1
        self.average = self.beta * self.average + (1 - self.beta) * value

    def compute(self):
        return self.average / (1 - self.beta**self.step)


class Mean(object):
    def __init__(self):
        self.values = []

    def compute(self):
        return sum(self.values) / len(self.values)

    def update(self, value):
        self.values.extend(np.reshape(value, [-1]))

    def reset(self):
        self.values = []

    def compute_and_reset(self):
        value = self.compute()
        self.reset()

        return value


def smooth(x, ksize=None):
    if ksize is None:
        ksize = len(x) // 10

        if ksize % 2 == 0:
            ksize += 1

    x = np.pad(x, (ksize // 2, ksize // 2), mode='reflect', reflect_type='odd')
    w = scipy.signal.windows.hann(ksize)
    w /= w.sum()
    x = scipy.signal.convolve(x, w, mode='valid')

    return x


def seed_python(seed):
    # os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    # tf.set_random_seed(seed)
    np.random.seed(seed)


def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_to_image():
    warnings.warn('use SummaryWriter.add_figure', DeprecationWarning)

    with tempfile.TemporaryFile() as f:
        plt.savefig(f)
        plt.close()
        image = Image.open(f)
        image = np.array(image)

        return image


def mkdir(path):
    os.makedirs(path, exist_ok=True)

    return path


def one_hot(input, num_classes):
    return torch.eye(num_classes).to(input.device)[input]
