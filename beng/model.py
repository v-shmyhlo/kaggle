import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class Model(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.encoder = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes, in_channels=1)
       
    def forward(self, input):
        return self.encoder(input)
