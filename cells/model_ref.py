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

        self.norm_i = nn.BatchNorm1d(embedding_size)
        self.norm_r = nn.BatchNorm1d(embedding_size)

        self.output = nn.Sequential(
            nn.Dropout(model.dropout),
            nn.Linear(embedding_size, num_classes))

    def forward(self, input, ref, feats, target=None):
        if self.training:
            assert target is not None
        else:
            assert target is None

        input = torch.cat([input, ref], 0)
        input = self.norm(input)
        input = self.model(input)
        input, ref = torch.split(input, input.size(0) // 2, 0)

        input = self.norm_i(input)
        ref = self.norm_r(ref)

        input = (input - ref) / torch.norm(ref, 2, 1)
        input = self.norm_o(input)

        output = self.output(input)

        return output
