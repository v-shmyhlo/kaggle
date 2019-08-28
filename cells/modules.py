import math

import torch
from torch import nn as nn
from torch.nn import functional as F

import utils


class SoftDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()

        self.p = p

    def forward(self, input):
        return soft_dropout(input, self.p, input.size(), training=self.training)


class SoftDropout2d(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()

        self.p = p

    def forward(self, input):
        return soft_dropout(input, self.p, (input.size(0), input.size(1), 1, 1), training=self.training)


def soft_dropout(input, p, noise_shape, training=True):
    if p < 0. or p > 1.:
        raise ValueError("soft_dropout probability has to be between 0 and 1, but got {}".format(p))

    if not training:
        return input

    weight = torch.empty(noise_shape, dtype=input.dtype, layout=input.layout, device=input.device) \
        .uniform_(1 - p, 1 + p)
    input = input * weight

    return input


class AdaptiveGeneralizedAvgPool2d(nn.Module):
    def __init__(self, output_size, p=3, eps=1e-6):
        super().__init__()

        self.output_size = output_size
        self.p = nn.Parameter(torch.tensor(p, dtype=torch.float))
        self.eps = eps

    def forward(self, input):
        input = input.clamp(min=self.eps)

        input = input.pow(self.p)
        input = F.adaptive_avg_pool2d(input, self.output_size)
        input = input.pow(1. / self.p)

        return input


class NormalizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))

        std = 1. / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -std, std)

    def forward(self, input):
        input = F.normalize(input, 2, 1)
        weight = F.normalize(self.weight, 2, 1)
        input = F.linear(input, weight)

        return input


# class ArcFace(nn.Module):
#     def __init__(self, num_classes, s=64., m=0.5):
#         super().__init__()
#
#         self.num_classes = num_classes
#         self.s = s
#         self.m = m
#
#     def forward(self, input, target):
#         if target is not None:
#             theta = torch.acos(input)
#             marginal_input = torch.cos(theta + self.m)
#
#             target_oh = utils.one_hot(target, self.num_classes)
#             input = (1 - target_oh) * input + target_oh * marginal_input
#
#         input = input * self.s
#
#         return input


class ArcFace(nn.Module):
    def __init__(self, s=30.0, m=0.5):
        super().__init__()

        self.s = s
        self.easy_margin = False
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, target):
        cosine = input
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = utils.one_hot(target, cosine.size(1))
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class ChannelReweight(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.weight = nn.Sequential(
            nn.BatchNorm1d(features * 2),
            nn.Linear(features * 2, 64),
            nn.PReLU(),
            nn.Linear(64, features),
            nn.Softplus())

    def forward(self, input):
        stats = torch.cat([
            input.mean((2, 3)),
            input.std((2, 3)),
        ], 1).detach()

        weight = self.weight(stats)
        weight = weight.view(weight.size(0), weight.size(1), 1, 1)

        input = input * weight

        return input
