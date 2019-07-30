import numbers
import random
from collections.abc import Iterable

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image

import transforms


class ApplyTo(object):
    def __init__(self, tos, transform):
        self.tos = tos
        self.transform = transform

    def __call__(self, input):
        input = {
            **input,
            **{to: self.transform(input[to]) for to in self.tos},
        }

        return input


class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            return hflip(image)

        if random.random() < self.p:
            return vflip(image)

        return image

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomTranspose(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            return transpose(image)

        return image

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image):
        return resize(image, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = torchvision.transforms.functional._pil_interpolation_to_str[self.interpolation]
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
    def get_params(image, output_size):
        w, h = image.size
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


class RandomRotation(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, image):
        angle = self.get_params(self.degrees)

        return rotate(image, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class ToTensor(object):
    def __call__(self, image):
        image = [F.to_tensor(c) for c in image]
        image = torch.cat(image, 0)

        return image

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomSite(object):
    def __call__(self, image):
        if random.random() < 0.5:
            return image[:6]
        else:
            return image[6:]


class SplitInSites(object):
    def __call__(self, image):
        return [image[:6], image[6:]]


class NormalizedColorJitter(object):
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, image):
        weight = torch.FloatTensor(image.size(0), 1, 1).uniform_(1 - self.weight, 1 + self.weight)
        weight = weight / weight.sum() * image.size(0)
        image = image * weight

        return image


class Extract(object):
    def __init__(self, fields):
        self.fields = fields

    def __call__(self, input):
        return tuple(input[k] for k in self.fields)


class StatColorJitter(object):
    def __init__(self):
        class_to_stats = torch.load('./stats.pth')
        class_to_stats = [s.view(*s.size(), 1, 1) for s in class_to_stats]

        self.class_to_stats = class_to_stats

    # TODO: check
    def __call__(self, input):
        stats = self.class_to_stats[input['label']]
        mean, std = stats[np.random.randint(stats.size(0))]
        dim = (1, 2)

        image = input['image']
        image = (image - image.mean(dim, keepdim=True)) / image.std(dim, keepdim=True).clamp(min=1e-7)
        image = image * std + mean

        return {
            **input,
            'image': image,
        }


class NormalizeByRefStats(object):
    def __call__(self, input):
        image, ref_stats = input['image'], input['ref_stats']

        ref_stats = ref_stats[np.random.randint(ref_stats.size(0))]
        mean, std = torch.split(ref_stats, 1, 1)
        mean, std = mean.view(mean.size(0), 1, 1), std.view(std.size(0), 1, 1)
        image = (image - mean) / std

        return {
            **input,
            'image': image
        }


class NormalizeByExperimentStats(object):
    def __init__(self, stats):
        self.stats = stats

    def __call__(self, input):
        mean, std = self.stats[input['exp']]
        mean, std = mean.view(mean.size(0), 1, 1), std.view(std.size(0), 1, 1)

        image = (input['image'] - mean) / std

        return {
            **input,
            'image': image
        }


class NormalizeByPlateStats(object):
    def __init__(self, stats):
        self.stats = stats

    def __call__(self, input):
        mean, std = self.stats[(input['exp'], input['plate'])]
        mean, std = mean.view(mean.size(0), 1, 1), std.view(std.size(0), 1, 1)

        image = (input['image'] - mean) / std

        return {
            **input,
            'image': image
        }


class TTA(object):
    def __call__(self, input):
        return [input, rotate(input, 90), rotate(input, 180), rotate(input, 270)]


# TODO: refactor

def resize(image, size, interpolation=Image.BILINEAR):
    return [F.resize(c, size, interpolation) for c in image]


def hflip(image):
    return [F.hflip(c) for c in image]


def vflip(image):
    return [F.vflip(c) for c in image]


def transpose(image):
    return [transforms.transpose(c) for c in image]


def rotate(image, angle, resample=False, expand=False, center=None):
    return [F.rotate(c, angle, resample, expand, center) for c in image]
