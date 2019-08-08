import efficientnet_pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()

        self.norm = nn.BatchNorm2d(6)
        self.features_norm = nn.BatchNorm1d(1280)

        if model.type.startswith('efficientnet'):
            self.model = efficientnet_pytorch.EfficientNet.from_pretrained(model.type)
            # self.model._conv_stem = efficientnet_pytorch.utils.Conv2dDynamicSamePadding(
            #     6, 32, kernel_size=3, stride=2, bias=False)
            self.model._conv_stem = nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.model._fc = nn.Linear(self.model._fc.in_features, num_classes * 4)
        else:
            raise AssertionError('invalid model {}'.format(model))

    def forward(self, input, feats, target=None):
        if self.training:
            assert target is not None
        else:
            assert target is None

        input = self.norm(input)

        input = self.model.extract_features(input)
        input = F.adaptive_avg_pool2d(input, 1).squeeze(-1).squeeze(-1)
        input = self.features_norm(input)

        if self.model._dropout:
            input = F.dropout(input, p=self.model._dropout, training=self.training)

        input = self.model._fc(input)
        input = input.view(input.size(0), 4, input.size(1) // 4)
        input = input[torch.arange(input.size(0)), feats[:, 0]]

        return input
