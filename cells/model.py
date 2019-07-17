import efficientnet_pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class NormalizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        input = F.normalize(input, 2, 1)
        weight = F.normalize(self.weight, 2, 1)
        input = F.linear(input, weight)

        return input


class ArcFace(nn.Module):
    def __init__(self, num_classes, s=64., m=0.5):
        super().__init__()

        self.num_classes = num_classes
        self.s = s
        self.m = m

    def forward(self, input, target):
        if target is not None:
            theta = torch.acos(input)
            marginal_input = torch.cos(theta + self.m)

            target_oh = utils.one_hot(target, self.num_classes)
            input = (1 - target_oh) * input + target_oh * marginal_input

        input = input * self.s

        return input


class Model(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()

        self.norm = nn.BatchNorm2d(6)

        self.model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')
        # self.model._conv_stem = efficientnet_pytorch.utils.Conv2dDynamicSamePadding(
        #     6, 32, kernel_size=3, stride=2, bias=False)
        self.model._conv_stem = nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model._dropout = model.dropout
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)

        self.output = nn.Sequential()

    def forward(self, input, feats, target=None):
        if self.training:
            assert target is not None
        else:
            assert target is None

        input = self.norm(input)
        input = self.model(input)
        output = self.output(input)

        return output
