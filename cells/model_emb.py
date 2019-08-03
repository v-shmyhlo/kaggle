import efficientnet_pytorch
import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F


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
        elif model.type == 'seresnext50':
            self.model = pretrainedmodels.se_resnext50_32x4d(pretrained='imagenet')
            self.model.layer0.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.model.last_linear = nn.Linear(self.model.last_linear.in_features, num_classes)
        else:
            raise AssertionError('invalid model {}'.format(model))

    def forward(self, input, feats, target=None):
        if self.training:
            assert target is not None
        else:
            assert target is None

        input = self.norm(input)

        # Convolution layers
        input = self.model.extract_features(input)

        # Pooling and final linear layer
        input = F.adaptive_avg_pool2d(input, 1).squeeze(-1).squeeze(-1)
        if self.model._dropout:
            input = F.dropout(input, p=self.model._dropout, training=self.training)
        embs = input
        input = self.model._fc(input)

        return input, embs
