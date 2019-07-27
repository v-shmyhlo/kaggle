import numbers
import random

import torchvision.transforms.functional as F


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

    def __call__(self, input):
        image = self.preprocess(input['image'])
        mask = self.preprocess(input['mask'])

        i, j, h, w = self.get_params(image, self.size)

        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        return {
            **input,
            'image': image,
            'mask': mask,
        }

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


class Extract(object):
    def __init__(self, fields):
        self.fields = fields

    def __call__(self, input):
        return tuple(input[k] for k in self.fields)
