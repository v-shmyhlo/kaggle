import numpy as np
import torchvision.transforms.functional as F


# TODO: check
class CentralCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, input):
        w, h = input.size
        i = 0
        j = (w - self.size) // 2
        w = self.size

        input = F.crop(input, i, j, h, w)

        return input


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, input):
        w, h = input.size
        i = 0
        j = np.random.randint(0, w - self.size + 1)
        w = self.size

        input = F.crop(input, i, j, h, w)

        return input
