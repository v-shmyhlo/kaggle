import pretrainedmodels
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()

        self.model = pretrainedmodels.se_resnet50(num_classes=1000, pretrained='imagenet')
        self.model.layer0.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, num_classes)
        self.norm = nn.BatchNorm2d(6)

    def forward(self, input):
        input = self.norm(input)
        input = self.model(input)

        return input
