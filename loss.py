import torch.nn as nn
import torch
import torch.nn.functional as F
from lovasz_losses import lovasz_hinge


# TODO: add reduction argument
# TODO: reduce same way everything


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.):
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

    tp = (target * input).sum(-1)
    # tn = ((1 - target) * (1 - input)).sum(-1)
    fp = ((1 - target) * input).sum(-1)
    fn = (target * (1 - input)).sum(-1)

    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)

    beta_sq = 2**2
    f2 = (1 + beta_sq) * p * r / (beta_sq * p + r + eps)
    loss = -(f2 + eps).log()

    return loss


def hinge_loss(input, target, delta=1.):
    positive_indices = (target > 0.5).float()
    negative_indices = (target <= 0.5).float()

    loss = 0.
    for i in range(input.size(0)):
        pos = positive_indices[i].nonzero()
        neg = negative_indices[i].nonzero()
        pos_examples = input[i, pos]
        neg_examples = torch.transpose(input[i, neg], 0, 1)
        loss += torch.sum(torch.max(torch.tensor(0.), delta + neg_examples - pos_examples))

    return loss


def bce_loss(input, target):
    loss = F.binary_cross_entropy_with_logits(input=input, target=target, reduction='sum')
    loss /= input.size(0)

    return loss


def lsep_loss(input, target):
    positive_indices = (target > 0.5).float()
    negative_indices = (target <= 0.5).float()

    loss = 0.
    for i in range(input.size(0)):
        pos = positive_indices[i].nonzero()
        neg = negative_indices[i].nonzero()
        pos_examples = input[i, pos]
        neg_examples = torch.transpose(input[i, neg], 0, 1)
        loss += torch.log(1 + torch.sum(torch.exp(neg_examples - pos_examples)))

    loss /= input.size(0)

    return loss


def lovasz_loss(input, target):
    loss = lovasz_hinge(logits=input, labels=target)
    loss = loss.mean()

    return loss
