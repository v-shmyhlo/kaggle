import torch.nn as nn


class Hswish()

class Bottleneck(nn.Module):
    def __init__(self, exp_channels, out_channels, kernel_size, stride, se, act):
        super().__init__()


class MobileNetV3(nn.Module):
    def __init__(self):
        super().__init__()

        relu = nn.ReLU()
        hswish = HSwish()

        self.conv = nn.Sequential(
            Bottleneck(16, 16, 3, stride=1, se=False, act=relu),
            Bottleneck(64, 24, 3, stride=2, se=False, act=relu),
            Bottleneck(72, 24, 3, stride=1, se=False, act=relu),
            Bottleneck(72, 40, 5, stride=2, se=True, act=relu),
            Bottleneck(120, 40, 5, sride=1, se=True, act=relu),
            Bottleneck(120, 40, 5, stride=1, se=True, act=relu),
            Bottleneck(240, 80, 3, stride=2, se=False, act=hswish),
            Bottleneck(200, 80, 3, stride=1, se=False, act=hswish),
            Bottleneck(184, 80, 3, stride=1, se=False, act=hswish),
            Bottleneck(184, 80, 3, stride=1, se=False, act=hswish),
            Bottleneck(480, 112, 3, stride=1, se=True, act=hswish),
            Bottleneck(672, 112, 3, stride=1, se=True, act=hswish),
            Bottleneck(672, 160, 5, stride=2, se=True, act=hswish),
            Bottleneck(960, 160, 5, stride=1, se=True, act=hswish),
            Bottleneck(960, 160, 5, stride=1, se=True, act=hswish),
        )
