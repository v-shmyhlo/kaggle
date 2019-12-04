import pytest
import torch

from detection.box_utils import boxes_area


@pytest.fixture()
def boxes():
    return torch.tensor([
        [0.1, 0.2, 0.2, 0.4],
        [10, 20, 20, 40],
        [0, 0, 5, 10],
    ])


def test_boxes_area(boxes):
    actual = boxes_area(boxes)
    expected = torch.tensor([
        0.02,
        200,
        50,
    ])

    assert torch.allclose(actual, expected)
