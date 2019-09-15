import efficientnet_pytorch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, model, num_classes, return_images=False, return_features=False):
        super().__init__()

        self.return_images = return_images
        self.return_features = return_features

        self.norm = nn.Sequential(
            nn.BatchNorm2d(6),
            # ChannelReweight(6),
        )

        if model.type.startswith('efficientnet'):
            self.model = efficientnet_pytorch.EfficientNet.from_pretrained(model.type)
            # self.model._conv_stem = efficientnet_pytorch.utils.Conv2dDynamicSamePadding(
            #     6, 32, kernel_size=3, stride=2, bias=False)
            self.model._conv_stem = nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)
        else:
            raise AssertionError('invalid model {}'.format(model))

    def forward(self, input, feats, target=None):
        if self.training:
            assert target is not None
        else:
            assert target is None

        input = self.norm(input)
        images = input
        input = self.model.extract_features(input)
        input = F.adaptive_avg_pool2d(input, 1).squeeze(-1).squeeze(-1)
        features = input
        if self.model._dropout:
            input = F.dropout(input, p=self.model._dropout, training=self.training)
        input = self.model._fc(input)

        return_values = (input,)
        if self.return_images:
            return_values = (*return_values, images)
        if self.return_features:
            return_values = (*return_values, features)

        if len(return_values) == 1:
            return return_values[0]

        return return_values
