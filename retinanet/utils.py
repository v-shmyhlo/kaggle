import torch


def boxes_tlbr_to_yxhw(boxes):
    t, l, b, r = torch.split(boxes, 1, 1)

    h = b - t
    w = r - l
    y = t + h / 2
    x = l + w / 2

    boxes = torch.cat([y, x, h, w], 1)

    return boxes


def boxes_yxhw_to_tlbr(boxes):
    y, x, h, w = torch.split(boxes, 1, 1)

    t = y - h / 2
    l = x - w / 2
    b = y + h / 2
    r = x + w / 2

    boxes = torch.cat([t, l, b, r], 1)

    return boxes
