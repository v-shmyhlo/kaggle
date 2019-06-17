import itertools
import math
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

BackboneOutput = namedtuple('BackboneOutput', ['c1', 'c2', 'c3', 'c4', 'c5'])
FPNOutput = namedtuple('FPNOutput', ['p3', 'p4', 'p5', 'p6', 'p7'])


# TODO: init

class ReLU(nn.ReLU):
    pass


class Norm(nn.GroupNorm):
    def __init__(self, num_features):
        super().__init__(num_channels=num_features, num_groups=32)


class ConvNorm(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2),
            Norm(out_channels))


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torchvision.models.resnet50(pretrained=True)

    def forward(self, input):
        input = self.model.conv1(input)
        input = self.model.bn1(input)
        input = self.model.relu(input)
        c1 = input
        input = self.model.maxpool(input)
        input = self.model.layer1(input)
        c2 = input
        input = self.model.layer2(input)
        c3 = input
        input = self.model.layer3(input)
        c4 = input
        input = self.model.layer4(input)
        c5 = input

        # input = self.avgpool(input)
        # input = input.view(input.size(0), -1)
        # input = self.fc(input)

        input = BackboneOutput(c1=c1, c2=c2, c3=c3, c4=c4, c5=c5)

        return input


class UpsampleMerge(nn.Module):
    def __init__(self, c_channels):
        super().__init__()

        self.projection = ConvNorm(c_channels, 256, 1)
        self.output = ConvNorm(256, 256, 3)

    def forward(self, p, c):
        # TODO: assert sizes

        p = F.interpolate(p, size=(c.size(2), c.size(3)), mode='nearest')
        c = self.projection(c)
        input = p + c
        input = self.output(input)

        return input


class FPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.c5_to_p6 = ConvNorm(2048, 256, 3, stride=2)
        self.p6_to_p7 = nn.Sequential(
            ReLU(inplace=True),
            ConvNorm(256, 256, 3, stride=2))
        self.c5_to_p5 = ConvNorm(2048, 256, 1)
        self.p5c4_to_p4 = UpsampleMerge(1024)
        self.p4c3_to_p3 = UpsampleMerge(512)

    def forward(self, input: BackboneOutput):
        p6 = self.c5_to_p6(input.c5)
        p7 = self.p6_to_p7(p6)
        p5 = self.c5_to_p5(input.c5)
        p4 = self.p5c4_to_p4(p5, input.c4)
        p3 = self.p4c3_to_p3(p4, input.c3)

        input = FPNOutput(p3=p3, p4=p4, p5=p5, p6=p6, p7=p7)

        return input


class HeadSubnet(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            ConvNorm(in_channels, in_channels, 3),
            ReLU(inplace=True),
            ConvNorm(in_channels, in_channels, 3),
            ReLU(inplace=True),
            ConvNorm(in_channels, in_channels, 3),
            ReLU(inplace=True),
            ConvNorm(in_channels, in_channels, 3),
            ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1))


class FlattenDetectionMap(nn.Module):
    def __init__(self, num_anchors):
        super().__init__()

        self.num_anchors = num_anchors

    def forward(self, input):
        b, c, h, w = input.size()
        input = input.view(b, c // self.num_anchors, self.num_anchors * h * w)
        input = input.permute(0, 2, 1)

        return input


class RetinaNet(nn.Module):
    def __init__(self, num_classes, num_anchors):
        super().__init__()

        self.backbone = Backbone()
        self.fpn = FPN()
        self.class_head = HeadSubnet(256, num_anchors * num_classes)
        self.regr_head = HeadSubnet(256, num_anchors * 4)
        self.flatten = FlattenDetectionMap(num_anchors)

        modules = itertools.chain(
            self.fpn.modules(),
            self.class_head.modules(),
            self.regr_head.modules())
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0., std=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        pi = 0.01
        nn.init.constant_(self.class_head[-1].bias, -math.log((1 - pi) / pi))

    def forward(self, input):
        backbone_output = self.backbone(input)
        fpn_output = self.fpn(backbone_output)

        class_output = torch.cat([self.flatten(self.class_head(x)) for x in fpn_output], 1)
        regr_output = torch.cat([self.flatten(self.regr_head(x)) for x in fpn_output], 1)

        return class_output, regr_output
