import numbers
import random

import cv2
import numpy as np
import torchvision.transforms.functional as F


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, input):
        if random.random() < self.p:
            return hflip(input)

        return input


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, input):
        if random.random() < self.p:
            return vflip(input)

        return input


class RandomTranspose(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, input):
        if random.random() < self.p:
            return transpose(input)

        return input


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
        h, w, _ = image.shape
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
        h, w, _ = image.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        m = mask[:, :, 1:].sum((0, 2))
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


def hflip(input):
    return {
        **input,
        'image': cv2.flip(input['image'], 1),
        'mask': cv2.flip(input['mask'], 1),
    }


def vflip(input):
    return {
        **input,
        'image': cv2.flip(input['image'], 0),
        'mask': cv2.flip(input['mask'], 0),
    }


def transpose(input):
    return {
        **input,
        'image': cv2.transpose(input['image']),
        'mask': cv2.transpose(input['mask']),
    }


# TODO: check
def crop(input, i, j, h, w):
    return {
        **input,
        'image': input['image'][i:i + h, j:j + w],
        'mask': input['mask'][i:i + h, j:j + w],
    }
