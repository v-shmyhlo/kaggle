import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class Model(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.norm = nn.BatchNorm2d(1)
        self.encoder = EfficientNet.from_pretrained(config.type, num_classes=num_classes, in_channels=1)

    def forward(self, input):
        input = self.norm(input)
        input = self.encoder(input)

        return input
