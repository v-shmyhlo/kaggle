import math

import torch
from torch import nn as nn
from torch.nn import functional as F

import utils


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


class SampleNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 4, channels, 1))

    def forward(self, input):
        unnormalized = input

        input = self.conv(input)
        mean, log_std = torch.split(input, input.size(1) // 2, 1)
        std = log_std.exp() + 1e-7

        print(input.shape, mean.shape, std.shape)

        input = (unnormalized - mean) / std

        return input
