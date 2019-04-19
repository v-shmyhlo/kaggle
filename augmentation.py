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
