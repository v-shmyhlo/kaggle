import torch.nn as nn


class HSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()

        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, input):
        input = self.relu6(input + 3) / 6

        return input


class HSwish(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()

        self.hsigmoid = HSigmoid(inplace=inplace)

    def forward(self, input):
        input = input * self.hsigmoid(input)

        return input


class Norm(nn.BatchNorm2d):
    def __init__(self, num_features):
        super().__init__(num_features, momentum=0.01)


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=True):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, groups=groups, bias=bias)


class ConvNorm(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super().__init__(
            Conv(in_channels, out_channels, kernel_size, stride=stride, groups=groups, bias=False),
            Norm(out_channels))


class SEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            Conv(channels // 4, channels, 1),
            HSigmoid(inplace=True))

    def forward(self, input):
        se = self.se(input)
        input = input * se

        return input


class Bottleneck(nn.Module):
    def __init__(self, in_channels, exp_channels, out_channels, kernel_size, stride, se, act):
        super().__init__()

        if se:
            se = SEBlock(exp_channels)
        else:
            se = nn.Sequential()

        self.conv = nn.Sequential(
            ConvNorm(in_channels, exp_channels, 1),
            act,
            ConvNorm(exp_channels, exp_channels, kernel_size, stride=stride, groups=exp_channels),
            se,
            act,
            ConvNorm(exp_channels, out_channels, 1))

        if in_channels == out_channels:
            self.identity = nn.Sequential()
        else:
            self.identity = None

    def forward(self, input):
        if self.identity is None:
            input = self.conv(input)
        else:
            input = self.conv(input) + self.identity(input)

        return input


class MobileNetV3(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        relu = nn.ReLU(inplace=True)
        hswish = HSwish(inplace=True)

        self.conv = nn.Sequential(
            ConvNorm(in_channels, 16, 3, stride=2),
            hswish,
            Bottleneck(16, 16, 16, 3, stride=1, se=False, act=relu),
            Bottleneck(16, 64, 24, 3, stride=2, se=False, act=relu),
            Bottleneck(24, 72, 24, 3, stride=1, se=False, act=relu),
            Bottleneck(24, 72, 40, 5, stride=2, se=True, act=relu),
            Bottleneck(40, 120, 40, 5, stride=1, se=True, act=relu),
            Bottleneck(40, 120, 40, 5, stride=1, se=True, act=relu),
            Bottleneck(40, 240, 80, 3, stride=2, se=False, act=hswish),
            Bottleneck(80, 200, 80, 3, stride=1, se=False, act=hswish),
            Bottleneck(80, 184, 80, 3, stride=1, se=False, act=hswish),
            Bottleneck(80, 184, 80, 3, stride=1, se=False, act=hswish),
            Bottleneck(80, 480, 112, 3, stride=1, se=True, act=hswish),
            Bottleneck(112, 672, 112, 3, stride=1, se=True, act=hswish),
            Bottleneck(112, 672, 160, 5, stride=2, se=True, act=hswish),
            Bottleneck(160, 960, 160, 5, stride=1, se=True, act=hswish),
            Bottleneck(160, 960, 160, 5, stride=1, se=True, act=hswish),
            ConvNorm(160, 960, 1),
            hswish,
            nn.AdaptiveAvgPool2d(1),
            Conv(960, 1280, 1),
            hswish,
            Conv(1280, num_classes, 1))

    def forward(self, input):
        input = self.conv(input)
        input = input.view(input.size(0), input.size(1))

        return input
