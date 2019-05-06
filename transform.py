import numpy as np
import torchvision.transforms.functional as F


class SquarePad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        max_side = np.max(img.size)
        left = (max_side - img.size[0]) // 2
        right = max_side - img.size[0] - left
        top = (max_side - img.size[1]) // 2
        bottom = max_side - img.size[1] - top

        return F.pad(img, (left, top, right, bottom), self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(fill={0}, padding_mode={1})'.format(self.fill, self.padding_mode)


class RatioPad(object):
    def __init__(self, ratio=(2 / 3, 3 / 2), fill=0, padding_mode='constant'):
        assert ratio[0] < ratio[1], '{} should be less than {}'.format(ratio[0], ratio[1])

        self.ratio = ratio
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        w, h = img.size
        ratio = w / h

        if ratio < self.ratio[0]:
            w = round(h * self.ratio[0])
        elif ratio > self.ratio[1]:
            h = round(w / self.ratio[1])

        left = (w - img.size[0]) // 2
        right = w - img.size[0] - left
        top = (h - img.size[1]) // 2
        bottom = h - img.size[1] - top

        return F.pad(img, (left, top, right, bottom), self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(ratio={0}, fill={1}, padding_mode={2})'.format(
            self.ratio, self.fill, self.padding_mode)
