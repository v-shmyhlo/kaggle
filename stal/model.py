import torch.nn as nn
import torch.nn.functional as F
import torchvision


# TODO: init

class ReLU(nn.ReLU):
    pass


class Norm(nn.BatchNorm2d):
    pass


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=True):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, groups=groups, bias=bias)


class ConvNorm(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super().__init__(
            Conv(in_channels, out_channels, kernel_size, stride=stride, groups=groups, bias=False),
            Norm(out_channels))


class UpsampleMerge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.bottom = nn.Sequential(
            ConvNorm(in_channels, out_channels, 1),
            ReLU(inplace=True))

        # self.refine = nn.Sequential(
        #     ConvNorm(out_channels, out_channels, 3),
        #     ReLU(inplace=True))

    def forward(self, bottom, left):
        bottom = self.bottom(bottom)
        bottom = F.interpolate(bottom, scale_factor=2, mode='bilinear')
        input = bottom + left
        # input = self.refine(input)

        return input


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torchvision.models.resnet18(pretrained=True)

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

        return [None, c1, c2, c3, c4, c5]


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        k = 1

        self.merge1 = UpsampleMerge(64 * k, 64)
        self.merge2 = UpsampleMerge(128 * k, 64 * k)
        self.merge3 = UpsampleMerge(256 * k, 128 * k)
        self.merge4 = UpsampleMerge(512 * k, 256 * k)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, fmaps):
        input = fmaps[5]

        input = self.merge4(input, fmaps[4])
        input = self.merge3(input, fmaps[3])
        input = self.merge2(input, fmaps[2])
        input = self.merge1(input, fmaps[1])

        return input


class Model(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()

        self.norm = nn.BatchNorm2d(3)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.output = Conv(64, num_classes, 1)

    def forward(self, input):
        input = self.norm(input)
        input = self.encoder(input)
        input = self.decoder(input)
        input = self.output(input)
        input = F.interpolate(input, scale_factor=2, mode='bilinear')

        return input
