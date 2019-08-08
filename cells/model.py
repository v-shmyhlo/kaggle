import efficientnet_pytorch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()

        self.norm = nn.BatchNorm2d(6)

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
        input = self.model(input)

        return input
