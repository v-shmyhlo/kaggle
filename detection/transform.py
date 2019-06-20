import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

from detection.anchors import build_anchors_maps
from detection.utils import boxes_tlbr_to_yxhw, boxes_yxhw_to_tlbr, encode_boxes


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, input):
        image, (class_ids, boxes, masks), maps = input

        w, h = image.size
        scale = self.size / min(w, h)
        w, h = round(w * scale), round(h * scale)

        image = image.resize((w, h), self.interpolation)
        boxes = boxes * scale
        masks = [m.resize((w, h), self.interpolation) for m in masks]

        return image, (class_ids, boxes, masks), maps


# TODO: test
class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, input):
        image, dets, maps = input

        w, h = image.size
        i = np.random.randint(0, h - self.size + 1)
        j = np.random.randint(0, w - self.size + 1)

        input = crop(input, (i, j), (self.size, self.size))

        return input


class MaskCropAndResize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, input):
        image, (class_ids, boxes, masks), maps = input

        masks = [m.crop((l.item(), t.item(), r.item(), b.item()))
                 for (t, l, b, r), m in zip(boxes_yxhw_to_tlbr(boxes), masks)]
        masks = [m.resize((self.size, self.size)) for m in masks]

        return image, (class_ids, boxes, masks), maps


class RandomFlipLeftRight(object):
    def __call__(self, input):
        if np.random.random() > 0.5:
            input = flip_left_right(input)

        return input


class ToTensor(object):
    def __call__(self, input):
        image, (class_ids, boxes, masks), maps = input
        image = F.to_tensor(image)

        if len(masks) == 0:
            masks = torch.zeros(0)
        else:
            masks = torch.stack([F.to_tensor(m) for m in masks], 0)

        return image, (class_ids, boxes, masks), maps


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, input):
        image, dets, maps = input
        image = F.normalize(image, mean=self.mean, std=self.std)

        return image, dets, maps


class BuildLabels(object):
    def __init__(self, anchors, p2, p7, min_iou, max_iou):
        self.anchors = anchors
        self.p2 = p2
        self.p7 = p7
        self.min_iou = min_iou
        self.max_iou = max_iou

    def __call__(self, input):
        image, dets, maps = input

        _, h, w = image.size()
        anchor_maps = build_anchors_maps((h, w), self.anchors, p2=self.p2, p7=self.p7)
        maps = encode_boxes(dets, anchor_maps, min_iou=self.min_iou, max_iou=self.max_iou)

        return image, dets, maps


# TODO: test
def flip_left_right(input):
    image, (class_ids, boxes, masks), maps = input

    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    w, _ = image.size
    boxes[:, 1] = w - boxes[:, 1]
    masks = [m.transpose(Image.FLIP_LEFT_RIGHT) for m in masks]

    return image, (class_ids, boxes, masks), maps


def denormalize(tensor, mean, std, inplace=False):
    if not inplace:
        tensor = tensor.clone()

    mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    return tensor


def crop(input, ij, hw):
    image, (class_ids, boxes, masks), maps = input

    i, j = ij
    h, w = hw

    image = image.crop((j, i, j + w, i + h))

    boxes = boxes_yxhw_to_tlbr(boxes)
    boxes[:, [0, 2]] -= i
    boxes[:, [1, 3]] -= j
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, h)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, w)
    boxes = boxes_tlbr_to_yxhw(boxes)
    masks = [m.crop((j, i, j + w, i + h)) for m in masks]

    keep = (boxes[:, 2] * boxes[:, 3]) >= 16**2
    class_ids = class_ids[keep]
    boxes = boxes[keep]
    masks = [m for m, k in zip(masks, keep) if k]
   
    return image, (class_ids, boxes, masks), maps
