import torch
from torch import nn as nn
from torch.nn import functional as F

from stal.model import Conv, ConvNorm, ReLU


# TODO: additional loss to predict number of positive pixels
# TODO: limit dataset to positive only
# TODO: plot classifier accuracy
# TODO: pl, select best samples
# TODO: bi-tempered loss
# TODO: grad accum
# TODO: deep supervision
# TODO: mobilenet
# TODO: better classification loss
# TODO: lsep for classification loss
# TODO: rrelu
# TODO: other radam impl
# TODO: aux classifiers (on different decoder sizes)
# TODO: train on downsampled masks
# TODO: deep supervision
# TODO: other decoders
# TODO: normalized focal (both for classifier and mask)
# TODO: sliding window classifier
# TODO: tta
# TODO: ranger (one more time)
# TODO: albu color augmentations
# TODO: plot crop distribution weights
# TODO: postprocessing
# TODO: pseudolabeling
# TODO: sample pos/neg in batch
# TODO: predict more than true/false for classification
# TODO: smaller max size
# TODO: effnet + swish hack
# TODO: scale aug
# TODO: openinng/closing aug
# TODO: postprocessing (dilation/erosion)
# TODO: edges weighting/label smoothing
# TODO: reweight loss
# TODO: mixup/cutmix
# TODO: cutout
# TODO: mixmatch
# TODO: mosaic
# TODO: strat by error
# TODO: augmentations
# TODO: postprocess: aproximate with polygon
# TODO: larger stride
# TODO: tversky loss
# TODO: other losses


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
