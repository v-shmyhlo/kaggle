import efficientnet_pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()

        assert model.type in ['b0', 'b1', 'b2']

        self.norm = nn.BatchNorm2d(6)

        self.model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-{}'.format(model.type))
        # self.model._conv_stem = efficientnet_pytorch.utils.Conv2dDynamicSamePadding(
        #     6, 32, kernel_size=3, stride=2, bias=False)
        self.model._conv_stem = nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)

    def forward(self, input, ref, feats, target=None):
        if self.training:
            assert target is not None
        else:
            assert target is None

        assert input.size(2) == input.size(3) == self.model._global_params.image_size

        input = torch.cat([input, ref], 0)
        input = self.norm(input)

        input = self.model.extract_features(input)
        input = F.adaptive_avg_pool2d(input, 1).squeeze(-1).squeeze(-1)
        if self.model._dropout:
            input = F.dropout(input, p=self.model._dropout, training=self.training)
        input = self.model._fc(input)

        input, ref = torch.split(input, input.size(0) // 2, 0)
        input = input - ref

        return input
