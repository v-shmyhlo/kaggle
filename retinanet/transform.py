import math

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image

MIN_IOU = 0.4
MAX_IOU = 0.5


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, input):
        image, (class_ids, boxes) = input

        w, h = image.size
        scale = self.size / min(w, h)
        w, h = round(w * scale), round(h * scale)

        image = image.resize((w, h), self.interpolation)
        boxes = boxes * scale

        return image, (class_ids, boxes)


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, input):
        image, (class_ids, boxes) = input

        w, h = image.size
        i = np.random.randint(0, h - self.size + 1)
        j = np.random.randint(0, w - self.size + 1)

        image = image.crop((j, i, j + self.size, i + self.size))

        boxes = boxes_yxhw_to_tlbr(boxes)
        boxes[:, [0, 2]] -= i
        boxes[:, [1, 3]] -= j
        boxes = boxes.clamp(0, self.size)
        boxes = boxes_to_tlbr_yxhw(boxes)
        boxes[:, 2:].clamp_(min=1.)  # FIXME:

        # TODO: fix keep
        # keep = (boxes[:, 2] * boxes[:, 3]) > 1
        # class_ids = class_ids[keep]
        # boxes = boxes[keep]

        return image, (class_ids, boxes)


class ToTensor(object):
    def __call__(self, input):
        image, (class_ids, boxes) = input
        image = F.to_tensor(image)

        return image, (class_ids, boxes)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, input):
        image, (class_ids, boxes) = input
        image = F.normalize(image, mean=self.mean, std=self.std)

        return image, (class_ids, boxes)


class BuildLabels(object):
    def __init__(self, anchors):
        self.anchors = anchors

    def __call__(self, input):
        image, (class_ids, boxes) = input

        # class_ids = torch.ones(10).long()
        # boxes = torch.ones(10, 4).float() * 100

        _, h, w = image.size()
        anchor_maps = build_anchors_maps((h, w), self.anchors)
        class_output, regr_output = assign_anchors((class_ids, boxes), anchor_maps)

        return image, (class_output, regr_output)


def assign_anchors(input, anchors):
    (class_ids, boxes) = input

    ious = boxes_iou(boxes, anchors)
    iou_values, iou_indices = ious.max(0)

    # build class_output
    class_output = class_ids[iou_indices] + 1
    class_output[iou_values < MIN_IOU] = 0
    class_output[(iou_values >= MIN_IOU) & (iou_values <= MAX_IOU)] = -1

    # build regr_output
    boxes = boxes[iou_indices]
    shifts = (boxes[:, :2] - anchors[:, :2]) / anchors[:, 2:]
    scales = boxes[:, 2:] / anchors[:, 2:]
    regr_output = torch.cat([shifts, scales.log()], 1)

    return class_output, regr_output


def boxes_iou(a, b):
    a = boxes_yxhw_to_tlbr(a)
    b = boxes_yxhw_to_tlbr(b)
    iou = torchvision.ops.box_iou(boxes_yxhw_to_tlbr(a), boxes_yxhw_to_tlbr(b))

    return iou


def boxes_yxhw_to_tlbr(boxes):
    y, x, h, w = torch.split(boxes, 1, 1)

    t = y - h / 2
    l = x - w / 2
    b = y + h / 2
    r = x + w / 2

    boxes = torch.cat([t, l, b, r], 1)

    return boxes


def boxes_to_tlbr_yxhw(boxes):
    t, l, b, r = torch.split(boxes, 1, 1)

    h = b - t
    w = r - l
    y = t + h / 2
    x = l + w / 2

    boxes = torch.cat([y, x, h, w], 1)

    return boxes


def build_anchors_maps(image_size, anchor_levels):
    h, w = image_size

    for _ in range(3):
        h, w = math.ceil(h / 2), math.ceil(w / 2)

    anchor_maps = []
    for anchors in anchor_levels:
        for anchor in anchors:
            anchor_map = build_anchor_map(image_size, (h, w), anchor)
            anchor_maps.append(anchor_map)
        h, w = math.ceil(h / 2), math.ceil(w / 2)

    anchor_maps = torch.cat(anchor_maps, 1).t()

    return anchor_maps


def build_anchor_map(image_size, map_size, anchor):
    cell_size = (image_size[0] / map_size[0], image_size[1] / map_size[1])

    y = torch.linspace(cell_size[0] / 2, image_size[0] - cell_size[0] / 2, map_size[0])
    x = torch.linspace(cell_size[1] / 2, image_size[1] - cell_size[1] / 2, map_size[1])

    y, x = torch.meshgrid(y, x)
    h = torch.ones(map_size) * anchor[0]
    w = torch.ones(map_size) * anchor[1]
    anchor_map = torch.stack([y, x, h, w])
    anchor_map = anchor_map.view(anchor_map.size(0), anchor_map.size(1) * anchor_map.size(2))

    return anchor_map
