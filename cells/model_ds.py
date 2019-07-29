import itertools

import efficientnet_pytorch
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

        self.conv_head1 = nn.Conv2d(40, 40 * 4, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(40 * 4)
        self.fc1 = nn.Linear(40 * 4, num_classes)
        self.conv_head2 = nn.Conv2d(112, 112 * 4, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(112 * 4)
        self.fc2 = nn.Linear(112 * 4, num_classes)

    def extract_features(self, input):
        # Stem
        input = efficientnet_pytorch.model.relu_fn(self.model._bn0(self.model._conv_stem(input)))

        # Blocks
        inputs = []
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            input = block(input, drop_connect_rate=drop_connect_rate)
            inputs.append(input)

        groups = itertools.groupby(inputs, key=lambda x: (x.size(2), x.size(3)))
        inputs = [list(v)[-1] for k, v in groups]
        inputs = inputs[-3:]

        layers = [
            (self.bn1, self.conv_head1),
            (self.bn2, self.conv_head2),
            (self.model._bn1, self.model._conv_head),
        ]

        # Head
        assert len(inputs) == len(layers)
        inputs = [efficientnet_pytorch.model.relu_fn(bn(conv(input))) for input, (bn, conv) in zip(inputs, layers)]

        return inputs

    def forward(self, input, feats, target=None):
        if self.training:
            assert target is not None
        else:
            assert target is None

        input = self.norm(input)

        # Convolution layers
        inputs = self.extract_features(input)

        # Pooling and final linear layer
        inputs = [F.adaptive_avg_pool2d(input, 1).squeeze(-1).squeeze(-1) for input in inputs]
        if self.model._dropout:
            inputs = [F.dropout(input, p=self.model._dropout, training=self.training) for input in inputs]

        layers = [self.fc1, self.fc2, self.model._fc]

        assert len(inputs) == len(layers)
        inputs = [fc(input) for input, fc in zip(inputs, layers)]

        return inputs
