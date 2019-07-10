import pretrainedmodels
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()

        self.norm = nn.BatchNorm2d(6)

        self.model = pretrainedmodels.resnet18(num_classes=1000, pretrained='imagenet')
        self.model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        embedding_size = self.model.last_linear.in_features
        self.model.last_linear = nn.Sequential()

        self.output = nn.Sequential(
            nn.Dropout(model.dropout),
            nn.Linear(embedding_size * 2, num_classes))

    def forward(self, input, ref, feats, target=None):
        if self.training:
            assert target is not None
        else:
            assert target is None

        input = torch.cat([input, ref], 0)
        input = self.norm(input)
        input = self.model(input)
        input, ref = torch.split(input, input.size(0) // 2)

        input = torch.cat([input, ref], 1)
        output = self.output(input)

        return output
