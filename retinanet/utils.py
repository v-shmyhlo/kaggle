import torch
import torchvision

MIN_IOU = 0.4
MAX_IOU = 0.5


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


def boxes_iou(a, b):
    a = boxes_yxhw_to_tlbr(a)
    b = boxes_yxhw_to_tlbr(b)
    iou = torchvision.ops.box_iou(boxes_yxhw_to_tlbr(a), boxes_yxhw_to_tlbr(b))

    return iou


def decode_boxes(input, anchors):
    class_output, regr_output = input

    # TODO:
    if class_output.dim() == 1:
        scores = torch.ones_like(class_output).float()
        class_ids = class_output - 1
        fg = class_output > 0
    else:
        scores, class_ids = class_output.max(1)
        fg = scores > 0.

    yx = regr_output[:, :2] * anchors[:, 2:] + anchors[:, :2]
    hw = regr_output[:, 2:].exp() * anchors[:, 2:]
    boxes = torch.cat([yx, hw], 1)

    boxes = boxes[fg]
    class_ids = class_ids[fg]
    scores = scores[fg]

    return class_ids, boxes, scores


def encode_boxes(input, anchors):
    class_ids, boxes = input

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
