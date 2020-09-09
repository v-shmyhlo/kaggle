import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet
from efficientnet_pytorch import EfficientNet


class ConvNorm(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size,
            stride=1, padding=0, dilation=1, groups=1,
            padding_mode='zeros'):
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
                bias=False, padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels))


class SimpleCNN(nn.Module):
    def __init__(self, in_channels, out_features):
        super().__init__()
        self.conv = nn.Sequential(
            ConvNorm(in_channels, 8, 3, stride=2, padding=3 // 2),
            nn.ReLU(inplace=True),
            ConvNorm(8, 16, 3, stride=2, padding=3 // 2),
            nn.ReLU(inplace=True),
            ConvNorm(16, 32, 3, stride=2, padding=3 // 2),
            nn.ReLU(inplace=True),
            ConvNorm(32, 64, 3, stride=2, padding=3 // 2),
            nn.ReLU(inplace=True),
            ConvNorm(64, 128, 3, stride=2, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1))
        self.linear = nn.Linear(128, out_features)

    def forward(self, input):
        input = self.conv(input)
        input = input.view(input.size(0), input.size(1))
        input = self.linear(input)

        return input


class STN(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = resnet.resnet18(pretrained=True)
        self.encoder.conv1 = nn.Conv2d(
            1, self.encoder.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, 6)
        nn.init.zeros_(self.encoder.fc.weight)
        bias = torch.tensor([
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=torch.float).view(6)
        self.encoder.fc.bias.data.copy_(bias)

        # self.encoder = SimpleCNN(1, 6)
        # nn.init.zeros_(self.encoder.linear.weight)
        # bias = torch.tensor([
        #     [1, 0, 0],
        #     [0, 1, 0]
        # ], dtype=torch.float).view(6)
        # self.encoder.linear.bias.data.copy_(bias)

    def forward(self, input):
        theta = self.encoder(input)
        # theta = torch.tensor([
        #     [1, 0, 0],
        #     [0, 1, 0]
        # ], dtype=input.dtype, device=input.device).view(1, 6).repeat(input.size(0), 1)
        theta = theta.view(theta.size(0), 2, 3)
        print(theta.mean(0))
        grid = F.affine_grid(theta, input.size())
        input = F.grid_sample(input, grid)

        return input


class Model(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.norm = nn.BatchNorm2d(1)
        # self.stn = STN()
        self.encoder = EfficientNet.from_pretrained(config.type, num_classes=num_classes, in_channels=1)

        # self.encoder = se_resnext50_32x4d(1000, pretrained='imagenet')
        # self.encoder.layer0.conv1 = nn.Conv2d(
        #     1, self.encoder.layer0.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        # self.encoder.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.encoder.last_linear = nn.Linear(self.encoder.last_linear.in_features, num_classes, bias=True)

    def forward(self, input):
        etc = {}

        input = self.norm(input)
        # input = self.stn(input)
        # etc['stn'] = input
        input = self.encoder(input)

        return input, etc
