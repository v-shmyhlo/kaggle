import os
import random
import tempfile
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import torch
from PIL import Image

from beng.train import weighted_sum


class RandomSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, size):
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return min(self.size, len(self.dataset))

    def __getitem__(self, item):
        return self.dataset[np.random.randint(len(self.dataset))]


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


class UpdatesPerSecond(Mean):
    def __init__(self):
        super().__init__()

        self.current = time.time()

    def update(self, value):
        current = time.time()
        delta = current - self.current
        super().update(value / delta)
        self.current = current


def label_smoothing(input, eps=0.1, dim=-1):
    return input * (1 - eps) + eps / input.size(dim)


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


def mixup(images_1, targets_1, alpha):
    indices = np.random.permutation(images_1.size(0))
    images_2, targets_2 = images_1[indices], targets_1[indices]

    lam = np.random.beta(alpha, alpha)
    lam = np.maximum(lam, 1 - lam)

    images = weighted_sum(images_1, images_2, lam)
    targets = weighted_sum(targets_1, targets_2, lam)

    return images, targets


# def cutmix(images_1, labels_1, alpha):
#     b, _, h, w = images_1.size()
#     perm = np.random.permutation(b)
#     images_2, labels_2 = images_1[perm], labels_1[perm]
# 
#     lam = np.random.beta(alpha, alpha)
#     lam = np.maximum(lam, 1 - lam)
# 
#     r_x = np.random.uniform(0, w)
#     r_y = np.random.uniform(0, h)
#     r_w = w * np.sqrt(1 - lam)
#     r_h = h * np.sqrt(1 - lam)
#     x1 = (r_x - r_w / 2).clip(0, w).round().astype(np.int32)
#     x2 = (r_x + r_w / 2).clip(0, w).round().astype(np.int32)
#     y1 = (r_y - r_h / 2).clip(0, h).round().astype(np.int32)
#     y2 = (r_y + r_h / 2).clip(0, h).round().astype(np.int32)
# 
#     images_1[:, :, x1:x2, y1:y2] = images_2[:, :, x1:x2, y1:y2]
#     images = images_1
#     labels = weighted_sum(labels_1, labels_2, lam)
# 
#     return images, labels


def cutmix(images_1, labels_1, alpha):
    b, _, h, w = images_1.size()
    perm = np.random.permutation(b)
    images_2, labels_2 = images_1[perm], labels_1[perm]

    lam = np.random.beta(alpha, alpha)
    lam = np.maximum(lam, 1 - lam)

    r_w = w * np.sqrt(1 - lam)
    r_h = h * np.sqrt(1 - lam)

    t = np.random.uniform(0, h - r_h)
    l = np.random.uniform(0, w - r_w)
    b = t + r_h
    r = l + r_w

    t, l, b, r = [np.round(p).astype(np.int32) for p in [t, l, b, r]]
    assert 0 <= t <= b <= h
    assert 0 <= l <= r <= w

    images_1[:, :, t:b, l:r] = images_2[:, :, t:b, l:r]
    images = images_1
    labels = weighted_sum(labels_1, labels_2, lam)

    return images, labels

# def cutmix(images_1, labels_1, alpha):
#     b, _, h, w = images_1.size()
#     perm = np.random.permutation(b)
#     images_2, labels_2 = images_1[perm], labels_1[perm]
#
#     for i in range(b):
#         lam = np.random.beta(alpha, alpha)
#         lam = np.maximum(lam, 1 - lam)
#
#         r_w = w * np.sqrt(1 - lam)
#         r_h = h * np.sqrt(1 - lam)
#
#         t = np.random.uniform(0, h - r_h)
#         l = np.random.uniform(0, w - r_w)
#         b = t + r_h
#         r = l + r_w
#
#         t, l, b, r = [np.round(p).astype(np.int32) for p in [t, l, b, r]]
#         assert 0 <= t <= b <= h
#         assert 0 <= l <= r <= w
#
#         images_1[i, :, t:b, l:r] = images_2[i, :, t:b, l:r]
#         labels_1[i] = weighted_sum(labels_1[i], labels_2[i], lam)
#
#     images = images_1
#     labels = labels_1
#
#     return images, labels
