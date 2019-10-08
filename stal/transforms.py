import numbers
import random

import numpy as np
import torchvision.transforms.functional as F

import transforms


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, input):
        if random.random() < self.p:
            return hflip(input)

        return input

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, input):
        if random.random() < self.p:
            return vflip(input)

        return input

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomTranspose(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, input):
        if random.random() < self.p:
            return transpose(input)

        return input

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


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
    def get_params(image, output_size):
        w, h = image.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, input):
        input = {
            **input,
            'image': self.preprocess(input['image']),
            'mask': self.preprocess(input['mask']),
        }

        i, j, h, w = self.get_params(input['image'], self.size)

        return crop(input, i, j, h, w)

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


class SampledRandomCrop(object):
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
    def get_params(image, mask, output_size):
        w, h = image.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        # TODO: no log
        m = (np.array(mask) != 0).sum(0)
        # m = np.log(np.e + m)
        m = m + 1

        m = np.convolve(m, np.ones(tw), mode='valid')
        m = m / m.sum()

        i = random.randint(0, h - th)
        j = np.random.choice(np.arange(0, w - tw + 1), p=m)

        return i, j, th, tw

    def __call__(self, input):
        input = {
            **input,
            'image': self.preprocess(input['image']),
            'mask': self.preprocess(input['mask']),
        }

        i, j, h, w = self.get_params(input['image'], input['mask'], self.size)

        return crop(input, i, j, h, w)

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


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, input):
        image = F.center_crop(input['image'], self.size)
        mask = F.center_crop(input['mask'], self.size)

        return {
            **input,
            'image': image,
            'mask': mask,
        }

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


def hflip(input):
    return {
        **input,
        'image': F.hflip(input['image']),
        'mask': F.hflip(input['mask']),
    }


def vflip(input):
    return {
        **input,
        'image': F.vflip(input['image']),
        'mask': F.vflip(input['mask']),
    }


def transpose(input):
    return {
        **input,
        'image': transforms.transpose(input['image']),
        'mask': transforms.transpose(input['mask']),
    }


def crop(input, i, j, h, w):
    return {
        **input,
        'image': F.crop(input['image'], i, j, h, w),
        'mask': F.crop(input['mask'], i, j, h, w),
    }
