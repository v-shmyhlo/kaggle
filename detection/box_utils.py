import torch
import torchvision


def boxes_tlbr_to_yxhw(boxes):
    t, l, b, r = torch.split(boxes, 1, -1)

    h = b - t
    w = r - l
    y = t + h / 2
    x = l + w / 2

    boxes = torch.cat([y, x, h, w], -1)

    return boxes


def boxes_yxhw_to_tlbr(boxes):
    y, x, h, w = torch.split(boxes, 1, -1)

    t = y - h / 2
    l = x - w / 2
    b = y + h / 2
    r = x + w / 2

    boxes = torch.cat([t, l, b, r], -1)

    return boxes


def boxes_iou(a, b):
    iou = torchvision.ops.box_iou(a, b)

    return iou


def boxes_area(boxes):
    tl, br = torch.split(boxes, 2, 1)
    hw = br - tl
    area = torch.prod(hw, 1)

    return area
