import torch

from detection.box_utils import boxes_area, boxes_iou


def test_boxes_iou():
    a = torch.tensor([
        [15, 20, 25, 40],
        [30, 20, 40, 40],
    ], dtype=torch.float)
    b = torch.tensor([
        [10, 20, 20, 40],
    ], dtype=torch.float)

    actual = boxes_iou(a, b)
    expected = torch.tensor([
        [1 / 3],
        [0.],
    ], dtype=torch.float)

    assert torch.allclose(actual, expected)


def test_boxes_area():
    boxes = torch.tensor([
        [0.1, 0.2, 0.2, 0.4],
        [10, 20, 20, 40],
        [0, 0, 5, 10],
    ], dtype=torch.float)

    actual = boxes_area(boxes)
    expected = torch.tensor([
        0.02,
        200,
        50,
    ], dtype=torch.float)

    assert torch.allclose(actual, expected)
