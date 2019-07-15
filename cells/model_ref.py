import efficientnet_pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()

        self.norm = nn.BatchNorm2d(6)

        self.model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')
        # self.model._conv_stem = efficientnet_pytorch.utils.Conv2dDynamicSamePadding(
        #     6, 32, kernel_size=3, stride=2, bias=False)
        self.model._conv_stem = nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model._dropout = model.dropout
        embedding_size = self.model._fc.in_features
        self.model._fc = nn.Linear(embedding_size, num_classes)

        self.norm_o = nn.BatchNorm1d(embedding_size)

    def forward(self, input, ref, feats, target=None):
        if self.training:
            assert target is not None
        else:
            assert target is None

        input = torch.cat([input, ref], 0)
        input = self.norm(input)
        input = self.model.extract_features(input)
        input = F.adaptive_avg_pool2d(input, 1).squeeze(-1).squeeze(-1)
        input, ref = torch.split(input, input.size(0) // 2, 0)

        input = input - ref
        input = F.normalize(input, 2, 1)
        input = self.norm_o(input)

        if self.model._dropout:
            input = F.dropout(input, p=self.model._dropout, training=self.training)
        input = self.model._fc(input)

        return input
