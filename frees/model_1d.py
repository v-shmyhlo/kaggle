import torch.nn as nn
import torch
import torchvision


def conv9(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3**2, stride=stride, padding=3**2 // 2, bias=False)


def conv1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ReLU(nn.RReLU):
    pass


class ResNet18MaxPool1d(nn.Module):
    def __init__(self, num_classes, dropout):
        super().__init__()

        self.model = ResNet1d(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.model.avgpool = nn.AdaptiveMaxPool1d(1)
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            self.model.fc)

        for m in self.model.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        input = input.unsqueeze(1)
        b, _, _ = input.size()

        input, images = self.model(input)
        images = images.unsqueeze(1)
        weights = torch.zeros(b, 1, 1, 1)

        return input, images, weights


class ResNet1d(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super().__init__()
        self.inplanes = 64

        self.layer0 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7**2, stride=2**2, padding=7**2 // 2, bias=False),
            nn.BatchNorm1d(64),
            ReLU(inplace=True))
        self.maxpool = nn.MaxPool1d(kernel_size=3**2, stride=2**2, padding=3**2 // 2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2**2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2**2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2**2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
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
                conv1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
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
        features = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, features


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv9(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = ReLU(inplace=True)
        self.conv2 = conv9(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
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
