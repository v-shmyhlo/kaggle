import numbers
import random
from collections.abc import Iterable

import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image


class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            return [F.hflip(c) for c in image]

        if random.random() < self.p:
            return [F.vflip(c) for c in image]

        return image

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image):
        return [F.resize(c, self.size, self.interpolation) for c in image]

    def __repr__(self):
        interpolate_str = torchvision.transforms._pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, image):
        return [F.center_crop(c, self.size) for c in image]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, image):
        image = [self.preprocess(c) for c in image]

        i, j, h, w = self.get_params(image[0], self.size)

        return [F.crop(c, i, j, h, w) for c in image]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

    def preprocess(self, image):
        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)

        return image


class ToTensor(object):
    def __call__(self, image):
        image = [F.to_tensor(c) for c in image]
        image = torch.cat(image, 0)

        return image

    def __repr__(self):
        return self.__class__.__name__ + '()'
