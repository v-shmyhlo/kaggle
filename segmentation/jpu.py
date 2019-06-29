import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# TODO: init

class ReLU(nn.ReLU):
    pass


class Norm(nn.BatchNorm2d):
    pass


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=(((kernel_size - 1) * dilation) + 1) // 2,
            dilation=dilation, groups=groups, bias=bias)


class ConvNorm(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
        super().__init__(
            Conv(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, groups=groups, bias=False),
            Norm(out_channels))


class Encoder(nn.Module):
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

        return [None, c1, c2, c3, c4, c5]


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        out_channels = 128

        # self.c1 = nn.Sequential(ConvNorm(64, 64, 3), ReLU(inplace=True))
        self.c2 = nn.Sequential(ConvNorm(256, out_channels, 3), ReLU(inplace=True))
        self.c3 = nn.Sequential(ConvNorm(512, out_channels, 3), ReLU(inplace=True))
        self.c4 = nn.Sequential(ConvNorm(1024, out_channels, 3), ReLU(inplace=True))
        self.c5 = nn.Sequential(ConvNorm(2048, out_channels, 3), ReLU(inplace=True))

        self.d1 = nn.Sequential(ConvNorm(out_channels * 4, out_channels, 3, dilation=1), ReLU(inplace=True))
        self.d2 = nn.Sequential(ConvNorm(out_channels * 4, out_channels, 3, dilation=2), ReLU(inplace=True))
        self.d4 = nn.Sequential(ConvNorm(out_channels * 4, out_channels, 3, dilation=4), ReLU(inplace=True))
        self.d8 = nn.Sequential(ConvNorm(out_channels * 4, out_channels, 3, dilation=8), ReLU(inplace=True))

    def forward(self, fmaps):
        base_scale = 1
        # c1 = self.c1(fmaps[1])
        c2 = F.interpolate(self.c2(fmaps[2]), scale_factor=base_scale * 1, mode='bilinear')
        c3 = F.interpolate(self.c3(fmaps[3]), scale_factor=base_scale * 2, mode='bilinear')
        c4 = F.interpolate(self.c4(fmaps[4]), scale_factor=base_scale * 4, mode='bilinear')
        c5 = F.interpolate(self.c5(fmaps[5]), scale_factor=base_scale * 8, mode='bilinear')

        input = torch.cat([c2, c3, c4, c5], 1)

        d1 = self.d1(input)
        d2 = self.d2(input)
        d4 = self.d4(input)
        d8 = self.d8(input)

        input = torch.cat([d1, d2, d4, d8], 1)

        return input


class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.output = Conv(512, num_classes, 1)

        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        input = self.encoder(input)
        input = self.decoder(input)
        input = self.output(input)
        input = F.interpolate(input, scale_factor=4, mode='bilinear')

        return input
