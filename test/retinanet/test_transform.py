import numpy as np
import torch
from PIL import Image

from retinanet.transform import crop
from retinanet.utils import boxes_tlbr_to_yxhw, boxes_yxhw_to_tlbr


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


def test_crop():
    image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    class_ids = torch.tensor([0], dtype=torch.long)
    boxes = torch.tensor([
        [30, 30, 20, 40],
    ], dtype=torch.float)

    image_new, (class_ids_new, boxes_new) = crop((image, (class_ids, boxes)), (30, 0), (20, 40))

    assert image_new.size == (40, 20)
    assert torch.equal(class_ids_new, torch.tensor([0], dtype=torch.long))
    assert torch.equal(boxes_new, torch.tensor([
        [5, 25, 10, 30]
    ], dtype=torch.float))
