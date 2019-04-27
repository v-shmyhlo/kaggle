import torch.nn as nn
import torchvision


class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        block = torchvision.models.resnet.Bottleneck
        self.model = torchvision.models.resnet.ResNet(block, [3, 4, 6, 3])
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, input):
        return self.model(input)
