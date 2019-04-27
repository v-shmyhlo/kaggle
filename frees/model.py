import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Model(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()

        if model == 'avg':
            self.model = AvgPoolModel(num_classes)
        elif model == 'attn':
            self.model = AttentionModel(num_classes)
        else:
            raise AssertionError('invalid model {}'.format(model))

    def forward(self, input):
        return self.model(input)


class AvgPoolModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        block = torchvision.models.resnet.BasicBlock
        self.model = torchvision.models.resnet.ResNet(block, [1, 1, 1, 1])
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, input):
        b, _, h, w = input.size()

        input = self.model(input)
        weights = torch.zeros(b, 1, h, w)

        return input, weights


class AttentionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        block = torchvision.models.resnet.BasicBlock
        self.model = torchvision.models.resnet.ResNet(block, [1, 1, 1, 1])
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model.fc = nn.Linear(512 * block.expansion, num_classes)

        self.weights = nn.Conv1d(512 * block.expansion, 1, kernel_size=1)

    def forward(self, input):
        _, _, h, w = input.size()

        input = self.model.conv1(input)
        input = self.model.bn1(input)
        input = self.model.relu(input)
        input = self.model.maxpool(input)

        input = self.model.layer1(input)
        input = self.model.layer2(input)
        input = self.model.layer3(input)
        input = self.model.layer4(input)

        input = input.mean(2)
        weights = self.weights(input)
        weights = weights.softmax(2)
        input = (input * weights).sum(2)

        input = self.model.fc(input)

        weights = weights.unsqueeze(2)
        # weights = F.interpolate(weights,size=(h, w), mode='nearest')  # TODO: speedup, mode
        weights = F.interpolate(weights, scale_factor=(4 * 4, 1 * 4), mode='nearest')  # TODO: speedup, mode

        return input, weights
