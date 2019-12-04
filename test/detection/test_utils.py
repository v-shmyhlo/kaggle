import torch

from detection.box_utils import boxes_tlbr_to_yxhw, boxes_yxhw_to_tlbr


def test_boxes_yxhw_to_tlbr():
    boxes = torch.tensor([
        [30, 30, 20, 40],
    ], dtype=torch.float)

    boxes = boxes_yxhw_to_tlbr(boxes)

    assert torch.equal(boxes, torch.tensor([
        [20, 10, 40, 50]
    ], dtype=torch.float))


def test_boxes_tlbr_to_yxhw():
    boxes = torch.tensor([
        [20, 10, 40, 50]
    ], dtype=torch.float)

    boxes = boxes_tlbr_to_yxhw(boxes)

    assert torch.equal(boxes, torch.tensor([
        [30, 30, 20, 40],
    ], dtype=torch.float))
