import torch
from torch import nn as nn
from torch.nn import functional as F

from stal.model import Conv, ConvNorm, ReLU


# TODO: resnet34
# TODO: plot crop distribution weights
# TODO: postprocessing
# TODO: transpose batch
# TODO: pseudolabeling
# TODO: crop sampling + image sampling
# TODO: effnet + swish hack
# TODO: kfold pseudolabeling
# TODO: scale aug
# TODO: dropcut
# TODO: openinng/closing aug
# TODO: label smoothing
# TODO: postprocessing (dilation/erosion)
# TODO: edges weighting/label smoothing
# TODO: reweight loss
# TODO: class-balanced sampler
# TODO: mixup/cutmix
# TODO: multi-layer class output


# TODO: check augs from kernels
# TODO: pseudolabeling
# TODO: mixmatch
# TODO: iterative stratification
# TODO: strat zeros separately
# TODO: mosaic
# TODO: strat by error
# TODO: augmentations
# TODO: postprocess: aproximate with polygon
# TODO: larger stride
# TODO: viz probs
# TODO: move mask.long() to transform
# TODO: lovasz


class SEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            Conv(channels // 4, channels, 1),
            nn.Sigmoid())

    def forward(self, input):
        se = self.se(input)
        input = input * se

        return input


class PyramidPooling(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.p2 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            ConvNorm(channels, channels // 4, 1),
            ReLU(inplace=True))
        self.p4 = nn.Sequential(
            nn.AvgPool2d(4, 4),
            ConvNorm(channels, channels // 4, 1),
            ReLU(inplace=True))
        self.p8 = nn.Sequential(
            nn.AvgPool2d(8, 8),
            ConvNorm(channels, channels // 4, 1),
            ReLU(inplace=True))
        self.pg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvNorm(channels, channels // 4, 1),
            ReLU(inplace=True))
        self.merge = nn.Sequential(
            ConvNorm(channels * 2, channels, 3),
            ReLU(inplace=True))

    def forward(self, input):
        p2 = F.interpolate(self.p2(input), scale_factor=2, mode='bilinear')
        p4 = F.interpolate(self.p4(input), scale_factor=4, mode='bilinear')
        p8 = F.interpolate(self.p8(input), scale_factor=8, mode='bilinear')
        pg = F.interpolate(self.pg(input), scale_factor=(input.size(2), input.size(3)), mode='bilinear')

        input = torch.cat([input, p2, p4, p8, pg], 1)
        input = self.merge(input)

        return input
