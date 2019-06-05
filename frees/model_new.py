import torch
import numpy as np
from gammatone.fftweight import fft_weights
import efficientnet_pytorch
from efficientnet_pytorch import EfficientNet
import torch.distributions
import torchvision
import librosa
import torch.nn as nn
import torch.nn.functional as F


class ReLU(nn.RReLU):
    pass


class Model(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()

        self.spectrogram = Spectrogram(model.sample_rate)

        if model.type == 'max':
            self.model = MaxPoolModel(num_classes, dropout=model.dropout)
        else:
            raise AssertionError('invalid model {}'.format(model.type))

    def forward(self, input):
        images = self.spectrogram(input)
        logits, weights = self.model(images)

        return logits, images, weights


class ConvNorm1d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super().__init__(
            nn.Conv1d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=False),
            nn.BatchNorm1d(out_channels))


class ConvNorm2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=False),
            nn.BatchNorm2d(out_channels))


class ConvNormRelu2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super().__init__(
            ConvNorm2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                groups=groups),
            ReLU(inplace=True))


class SplitConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.a = nn.Sequential(
            ConvNormRelu2d(in_channels, out_channels // 2, (7, 1), stride=(2, 1), padding=(3, 0)),
            ConvNormRelu2d(out_channels // 2, out_channels // 2, (1, 7), stride=(1, 2), padding=(0, 3)))
        self.b = nn.Sequential(
            ConvNormRelu2d(in_channels, out_channels // 2, (1, 7), stride=(1, 2), padding=(0, 3)),
            ConvNormRelu2d(out_channels // 2, out_channels // 2, (7, 1), stride=(2, 1), padding=(3, 0)))

    def forward(self, input):
        a = self.a(input)
        b = self.b(input)
        input = torch.cat([a, b], 1)

        return input


# class CustomBlock(nn.Sequential):
#     def __init__(self, in_features, out_features):
#         super().__init__(
#             ConvNormRelu2d(in_features, out_features, 3, 1, 1),
#             ConvNormRelu2d(out_features, out_features, 3, 1, 1),
#             nn.MaxPool2d(2, 2))

# class CustomBlock(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=5):
#         super().__init__()
#
#         padding = kernel_size // 2
#
#         self.a = nn.Sequential(
#             ConvNormRelu2d(in_channels, out_channels // 2, (kernel_size, 1), stride=(2, 1), padding=(padding, 0)),
#             ConvNormRelu2d(out_channels // 2, out_channels // 2, (1, kernel_size), stride=(1, 2), padding=(0, padding)))
#         self.b = nn.Sequential(
#             ConvNormRelu2d(in_channels, out_channels // 2, (1, kernel_size), stride=(1, 2), padding=(0, padding)),
#             ConvNormRelu2d(out_channels // 2, out_channels // 2, (kernel_size, 1), stride=(2, 1), padding=(padding, 0)))
#
#     def forward(self, input):
#         a = self.a(input)
#         b = self.b(input)
#         input = torch.cat([a, b], 1)
#
#         return input


# class CustomBlock(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
#         super().__init__()
#
#         padding = kernel_size // 2
#
#         self.a = nn.Sequential(
#             ConvNormRelu2d(
#                 in_channels, out_channels // 2, (kernel_size, 1), stride=(stride, 1), padding=(padding, 0)),
#             ConvNormRelu2d(
#                 out_channels // 2, out_channels // 2, (1, kernel_size), stride=(1, stride), padding=(0, padding)))
#         self.b = nn.Sequential(
#             ConvNormRelu2d(
#                 in_channels, out_channels // 2, (1, kernel_size), stride=(1, stride), padding=(0, padding)),
#             ConvNormRelu2d(
#                 out_channels // 2, out_channels // 2, (kernel_size, 1), stride=(stride, 1), padding=(padding, 0)))
#
#         self.identity = None
#
#     def forward(self, input):
#         a = self.a(input)
#         b = self.b(input)
#         input = torch.cat([a, b], 1)
#
#         return input

# class CustomBlock(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
#         super().__init__()
#
#         padding = kernel_size // 2
#
#         self.a = nn.Sequential(
#             ConvNormRelu2d(in_channels, out_channels, (kernel_size, 1), stride=(stride, 1), padding=(padding, 0)),
#             ConvNormRelu2d(out_channels, out_channels, (1, kernel_size), stride=(1, stride), padding=(0, padding)))
#         self.b = nn.Sequential(
#             ConvNormRelu2d(in_channels, out_channels, (1, kernel_size), stride=(1, stride), padding=(0, padding)),
#             ConvNormRelu2d(out_channels, out_channels, (kernel_size, 1), stride=(stride, 1), padding=(padding, 0)))
#         self.w = nn.Sequential(
#             ConvNorm2d(in_channels, out_channels, 3, stride=stride, padding=1),
#             nn.Sigmoid())
#
#     def forward(self, input):
#         a = self.a(input)
#         b = self.b(input)
#         w = self.w(input)
#         input = w * a + (1 - w) * b
#
#         return input


# class CustomBlock(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
#         super().__init__()
# 
#         padding = kernel_size // 2
# 
#         self.a = nn.Sequential(
#             ConvNormRelu2d(in_channels, out_channels, (kernel_size, 1), stride=(stride, 1), padding=(padding, 0)),
#             ConvNorm2d(out_channels, out_channels, (1, kernel_size), stride=(1, stride), padding=(0, padding)))
#         self.b = nn.Sequential(
#             ConvNormRelu2d(in_channels, out_channels, (1, kernel_size), stride=(1, stride), padding=(0, padding)),
#             ConvNorm2d(out_channels, out_channels, (kernel_size, 1), stride=(stride, 1), padding=(padding, 0)))
#         self.w = nn.Sequential(
#             nn.MaxPool2d(kernel_size, stride=stride, padding=padding),
#             ConvNormRelu2d(in_channels, out_channels // 4, 1),
#             ConvNorm2d(out_channels // 4, out_channels, 1),
#             nn.Sigmoid())
#         self.relu = ReLU(inplace=True)
# 
#     def forward(self, input):
#         a = self.a(input)
#         b = self.b(input)
#         w = self.w(input)
#         input = w * a + (1 - w) * b
#         input = self.relu(input)
# 
#         return input


# class CustomBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1):
#         super().__init__()
#
#         self.input = nn.Sequential(
#             ConvNorm2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2),
#             ReLU(inplace=True))
#
#         if in_channels == out_channels:
#             self.identity = nn.Sequential()
#         else:
#             self.identity = ConvNorm2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
#
#         self.relu = ReLU(inplace=True)
#
#     def forward(self, input):
#         input = self.input(input) + self.identity(input)
#         input = self.relu(input)
#
#         return input


# class CustomBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size):
#         super().__init__()
#
#         self.identity = nn.Sequential(
#             ConvNorm2d(in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2))
#         self.input = nn.Sequential(
#             ConvNorm2d(in_channels, out_channels // 4, 1),
#             ReLU(inplace=True),
#             ConvNorm2d(out_channels // 4, out_channels // 4, kernel_size, stride=2, padding=kernel_size // 2),
#             ReLU(inplace=True),
#             ConvNorm2d(out_channels // 4, out_channels, 1))
#         self.relu = ReLU(inplace=True)
#
#     def forward(self, input):
#         identity = self.identity(input)
#         input = self.input(input)
#
#         input = input + identity
#         input = self.relu(input)
#
#         return input

class CustomBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        kernel_size = 3

        self.identity = nn.Sequential(
            ConvNorm2d(in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2))
        self.input = nn.Sequential(
            ConvNorm2d(in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2),
            ReLU(inplace=True),
            ConvNorm2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2))
        self.relu = ReLU(inplace=True)

    def forward(self, input):
        identity = self.identity(input)
        input = self.input(input)

        input = input + identity
        input = self.relu(input)

        return input


class HeightPool(nn.Module):
    def forward(self, input):
        input = F.max_pool2d(input, (input.size(2), 1))
        input = input.squeeze(2)

        return input


class SelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.q = ConvNorm1d(in_channels, out_channels, 1)
        self.k = ConvNorm1d(in_channels, out_channels, 1)
        self.v = ConvNorm1d(in_channels, out_channels, 1)

    def forward(self, input):
        q = self.q(input)
        k = self.k(input)
        v = self.v(input)

        w = torch.bmm(q.permute(0, 2, 1), k)

        v = v.unsqueeze(2)
        w = w.unsqueeze(1)

        c = v * w.softmax(3)
        c = c.sum(3)

        return c


class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.v = ConvNorm1d(channels, channels, 1)
        self.w = ConvNorm1d(channels, 1, 1)

    def forward(self, input):
        v = self.v(input)
        w = self.w(input)

        c = v * w.softmax(2)
        c = c.sum(2)

        return c


class CustomModel(nn.Module):
    def __init__(self, num_classes, dropout):
        super().__init__()

        channels = 16

        self.blocks = nn.Sequential(
            ConvNorm2d(1, channels * 1, 3, padding=3 // 2),
            nn.ReLU(inplace=True),
            CustomBlock(channels * 1, channels * 2),
            CustomBlock(channels * 2, channels * 4),
            CustomBlock(channels * 4, channels * 8),
            CustomBlock(channels * 8, channels * 16),
            CustomBlock(channels * 16, channels * 32))
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.output = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Linear(channels * 32, num_classes))

        assert self.output[1].in_features == 512

    def forward(self, input):
        input = self.blocks(input)
        assert input.size(2) == 128 / (2**5)

        input = self.pool(input)
        input = input.view(input.size(0), input.size(1))
        input = self.output(input)

        return input


class MaxPoolModel(nn.Module):
    def __init__(self, num_classes, dropout):
        super().__init__()

        # self.model = pretrainedmodels.resnet18(pretrained=None)
        # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.model.avgpool = nn.AdaptiveMaxPool2d(1)
        # self.model.last_linear = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(512, num_classes))

        # self.model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        # # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.model.layer0 = SplitConv(1, 64)
        # self.model.avgpool = nn.AdaptiveMaxPool2d(1)
        # self.model.fc = nn.Sequential(
        #     nn.Dropout(dropout),
        #     self.model.fc)

        self.model = CustomModel(num_classes, dropout)

        for m in self.model.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=(1 / 8 + 1 / 3) / 2)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        b, _, h, w = input.size()

        input = self.model(input)
        weights = torch.zeros(b, 1, h, w)

        return input, weights


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super().__init__()
        self.inplanes = 64

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, torchvision.models.resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                torchvision.models.resnet.conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = torchvision.models.resnet.conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = ReLU(inplace=True)
        self.conv2 = torchvision.models.resnet.conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Spectrogram(nn.Module):
    def __init__(self, rate):
        super().__init__()

        self.n_fft = round(0.025 * rate)
        self.hop_length = round(0.01 * rate)

        filters = librosa.filters.mel(rate, self.n_fft)
        # filters, _ = fft_weights(self.n_fft, rate, 128, width=1, fmin=0, fmax=rate / 2, maxlen=self.n_fft / 2 + 1)

        filters = filters.reshape((*filters.shape, 1))
        filters = torch.tensor(filters).float()

        self.mel = nn.Conv1d(512, 128, 1, bias=False)
        self.mel.weight.data = filters
        self.mel.weight.requires_grad = False

        self.norm = nn.BatchNorm2d(1)

    def forward(self, input):
        input = torch.stft(input, n_fft=self.n_fft, hop_length=self.hop_length)
        input = torch.norm(input, 2, -1)**2  # TODO:
        input = self.mel(input)

        amin = torch.tensor(1e-10).to(input.device)
        input = 10.0 * torch.log10(torch.max(amin, input))

        input = input.unsqueeze(1)
        input = self.norm(input)

        return input
