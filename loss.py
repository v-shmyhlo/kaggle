import torch.nn as nn
import torch
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        target = target.float()
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size()) == 2:
            loss = loss.sum(dim=1)
        return loss.mean()


def f2_loss(input, target, eps=1e-7):
    input = input.sigmoid()

    tp = (target * input).sum(1)
    # tn = ((1 - target) * (1 - input)).sum(1)
    fp = ((1 - target) * input).sum(1)
    fn = (target * (1 - input)).sum(1)

    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)

    beta_sq = 2**2
    f2 = (1 + beta_sq) * p * r / (beta_sq * p + r + eps)
    loss = -(f2 + eps).log()

    return loss


def hinge_loss(input, target, delta=1.):
    target = target * 1. + (1 - target) * -1.
    loss = torch.max(torch.zeros_like(input).to(input.device), delta - target * input)

    return loss
