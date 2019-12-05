import torch


def logit(input):
    return torch.log(input / (1 - input))
