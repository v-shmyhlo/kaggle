import torch
from utils import one_hot


def test_one_hot():
    input = torch.tensor([1, 2, 0], dtype=torch.long)

    assert torch.equal(one_hot(input, 3), torch.tensor([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ], dtype=torch.float))
   