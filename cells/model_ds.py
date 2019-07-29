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

    def extract_features(self, input):
        # Stem
        input = efficientnet_pytorch.model.relu_fn(self.model._bn0(self.model._conv_stem(input)))

        # Blocks
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            input = block(input, drop_connect_rate=drop_connect_rate)
            print(input.shape)

        # Head
        input = efficientnet_pytorch.model.relu_fn(self.model._bn1(self.model._conv_head(input)))

        return input

    def forward(self, input, feats, target=None):
        if self.training:
            assert target is not None
        else:
            assert target is None

        input = self.norm(input)

        # Convolution layers
        input = self.extract_features(input)

        # Pooling and final linear layer
        input = F.adaptive_avg_pool2d(input, 1).squeeze(-1).squeeze(-1)
        if self.model._dropout:
            input = F.dropout(input, p=self.model._dropout, training=self.training)
        input = self.model._fc(input)

        fail

        return input
