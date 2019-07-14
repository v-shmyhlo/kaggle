import efficientnet_pytorch
import numpy as np
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
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)

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
        features = input
        if self.model._dropout:
            input = F.dropout(input, p=self.model._dropout, training=self.training)
        input = self.model._fc(input)

        return input, features


def get_shuffle_indices(target):
    eq = target.unsqueeze(1) == target.unsqueeze(0)
    eq[torch.eye(target.size(0)).byte()] = 0
    indices = [np.where(row)[0] for row in eq.data.cpu().numpy()]
    indices = [np.random.choice(row) if row.shape[0] > 0 else i for i, row in enumerate(indices)]
    indices = torch.tensor(indices).to(target.device)

    return indices
