import torch

from metric import accuracy


def test_accuracy():
    input = torch.tensor([
        [2, 3, 1],
        [5, 7, 0],
        [0, 9, -1]
    ], dtype=torch.float)
    target = torch.tensor([
        1,
        0,
        2
    ], dtype=torch.long)

    assert torch.equal(accuracy(input=input, target=target, topk=1), torch.tensor([
        1.,
        0.,
        0.
    ]))

    assert torch.equal(accuracy(input=input, target=target, topk=2), torch.tensor([
        1.,
        1.,
        0.
    ]))
