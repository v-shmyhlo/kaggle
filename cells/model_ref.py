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
        embedding_size = self.model._fc.in_features
        self.model._fc = nn.Linear(512, num_classes)

        self.input_output = nn.Sequential(
            nn.Linear(embedding_size, 512),
            nn.BatchNorm1d(512))
        self.ref_output = nn.Sequential(
            nn.Linear(embedding_size, 1024),
            nn.BatchNorm1d(1024))

    def forward(self, input, ref, feats, target=None):
        if self.training:
            assert target is not None
        else:
            assert target is None

        input = torch.cat([input, ref], 0)
        input = self.norm(input)

        # Convolution layers
        input = self.model.extract_features(input)

        # Pooling and final linear layer
        input = F.adaptive_avg_pool2d(input, 1).squeeze(-1).squeeze(-1)

        input, ref = torch.split(input, input.size(0) // 2, 0)

        input = self.input_output(input)
        ref = self.ref_output(ref)

        mean, log_var = torch.split(ref, ref.size(1) // 2, 1)
        std = torch.exp(0.5 * log_var)

        input = (input - mean) / std

        if self.model._dropout:
            input = F.dropout(input, p=self.model._dropout, training=self.training)
        input = self.model._fc(input)

        return input
