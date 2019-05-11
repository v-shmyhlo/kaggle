import numpy as np
import torch
import torchvision.transforms.functional as F


# TODO: refactor
# TODO: repr

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


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        _, h, w = img.size()

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(-self.length // 2, h + self.length // 2)
            x = np.random.randint(-self.length // 2, w + self.length // 2)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
