import torch
import torchvision

from detection.utils import boxes_iou, boxes_yxhw_to_tlbr


def encode_boxes(input, anchors, min_iou, max_iou):
    class_ids, boxes = input

    if boxes.size(0) == 0:
        class_output = torch.zeros(anchors.size(0), dtype=torch.long)
        regr_output = torch.zeros(anchors.size(0), 4, dtype=torch.float)

        return class_output, regr_output

    ious = boxes_iou(boxes, anchors)
    iou_values, iou_indices = ious.max(0)

    # build class_output
    class_output = class_ids[iou_indices] + 1
    class_output[iou_values < min_iou] = 0
    class_output[(iou_values >= min_iou) & (iou_values <= max_iou)] = -1

    # build regr_output
    boxes = boxes[iou_indices]
    shifts = (boxes[:, :2] - anchors[:, :2]) / anchors[:, 2:]
    scales = boxes[:, 2:] / anchors[:, 2:]
    regr_output = torch.cat([shifts, scales.log()], 1)

    return class_output, regr_output


def decode_boxes(input, anchors):
    class_output, regr_output = input

    scores, class_ids = class_output.max(1)
    fg = scores > 0.

    yx = regr_output[:, :2] * anchors[:, 2:] + anchors[:, :2]
    hw = regr_output[:, 2:].exp() * anchors[:, 2:]
    boxes = torch.cat([yx, hw], 1)

    boxes = boxes[fg]
    class_ids = class_ids[fg]
    scores = scores[fg]

    keep = torchvision.ops.nms(boxes_yxhw_to_tlbr(boxes), scores, 0.5)
    boxes = boxes[keep]
    class_ids = class_ids[keep]
    scores = scores[keep]

    return class_ids, boxes, scores
