import torch
# from gammatone.fftweight import fft_weights
import torch.distributions
import torchvision
import librosa
import torch.nn as nn
from frees.spec_augment import spec_augment
from frees.model_1d import ResNet18MaxPool1d


class ReLU(nn.RReLU):
    pass


class Model(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()

        self.model_type = model.type

        if model.type == 'resnet18-maxpool-2d':
            self.spectrogram = Spectrogram(model.sample_rate)
            self.model = ResNet18MaxPool2d(num_classes, dropout=model.dropout)
        elif model.type == 'mobnetv2-maxpool-2d':
            # TODO: dropout, pretrained
            self.spectrogram = Spectrogram(model.sample_rate)
            self.model = torchvision.models.mobilenet_v2(num_classes=num_classes)
            self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        elif model.type == 'resnet18-maxpool-1d':
            self.model = ResNet18MaxPool1d(num_classes, dropout=model.dropout)
        else:
            raise AssertionError('invalid model {}'.format(model.type))

    def forward(self, input, spec_aug=False):
        if self.model_type == 'resnet18-maxpool-2d':
            images = self.spectrogram(input, spec_aug=spec_aug)
            logits, weights = self.model(images)
        elif self.model_type == 'mobnetv2-maxpool-2d':
            images = self.spectrogram(input, spec_aug=spec_aug)
            logits = self.model(images)
            weights = torch.zeros(logits.size(0), 1, 1, 1)
        elif self.model_type == 'resnet18-maxpool-1d':
            logits, images, weights = self.model(input)
        else:
            raise AssertionError('invalid model {}'.format(self.model_type))

        return logits, images, weights


class ConvNormRelu2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            ReLU(inplace=True))


class SplitConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        out_channels = out_channels // 2
        kernel_size = 7
        padding = kernel_size // 2

        self.a = nn.Sequential(
            ConvNormRelu2d(in_channels, out_channels, (kernel_size, 1), stride=(2, 1), padding=(padding, 0)),
            ConvNormRelu2d(out_channels, out_channels, (1, kernel_size), stride=(1, 2), padding=(0, padding)))
        self.b = nn.Sequential(
            ConvNormRelu2d(in_channels, out_channels, (1, kernel_size), stride=(1, 2), padding=(0, padding)),
            ConvNormRelu2d(out_channels, out_channels, (kernel_size, 1), stride=(2, 1), padding=(padding, 0)))

    def forward(self, input):
        a = self.a(input)
        b = self.b(input)
        input = torch.cat([a, b], 1)

        return input


class ResNet18MaxPool2d(nn.Module):
    def __init__(self, num_classes, dropout):
        super().__init__()

        self.model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.model.layer0 = SplitConv(1, 64)
        self.model.avgpool = nn.AdaptiveMaxPool2d(1)
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            self.model.fc)

        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
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
        self.mel = self.filters_to_conv(filters)

        # filters, _ = fft_weights(self.n_fft, rate, 128, width=1, fmin=0, fmax=rate / 2, maxlen=self.n_fft / 2 + 1)
        # self.gamma = self.filters_to_conv(filters)

        # self.coord = HeightCoord(128)
        self.norm = nn.BatchNorm2d(1)

    def forward(self, input, spec_aug):
        input = torch.stft(input, n_fft=self.n_fft, hop_length=self.hop_length)
        input = torch.norm(input, 2, -1)**2  # TODO:

        input = torch.stack([
            self.mel(input),
            # self.gamma(input)
        ], 1)
        amin = torch.tensor(1e-10).to(input.device)
        input = 10.0 * torch.log10(torch.max(amin, input))

        # input = self.coord(input)
        input = self.norm(input)

        if self.training and spec_aug:
            for i in range(input.shape[0]):
                spec_augment(input[i])

        return input

    @staticmethod
    def filters_to_conv(filters):
        filters = filters.reshape((*filters.shape, 1))
        filters = torch.tensor(filters).float()

        conv = nn.Conv1d(1, 1, 1, bias=False)
        conv.weight.data = filters
        conv.weight.requires_grad = False

        return conv


class HeightCoord(nn.Module):
    def __init__(self, height):
        super().__init__()

        self.coord = nn.Parameter(torch.linspace(-1, 1, height).view(1, 1, height, 1))
        self.coord.requires_grad = False

    def forward(self, input):
        b, _, h, w = input.size()

        coord = torch.ones(b, 1, h, w).to(input.device)
        coord = coord * self.coord
        input = torch.cat([input, coord], 1)

        return input
