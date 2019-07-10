import numpy as np
import pretrainedmodels
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


# TODO: norm layers


class NormalizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        input = F.normalize(input, 2, 1)
        weight = F.normalize(self.weight, 2, 1)
        input = F.linear(input, weight)

        return input


class ArcFace(nn.Module):
    def __init__(self, num_classes, s=64., m=0.5):
        super().__init__()

        self.num_classes = num_classes
        self.s = s
        self.m = m

    def forward(self, input, target):
        if target is not None:
            theta = torch.acos(input)
            marginal_input = torch.cos(theta + self.m)

            target_oh = utils.one_hot(target, self.num_classes)
            input = (1 - target_oh) * input + target_oh * marginal_input

        input = input * self.s

        return input


class Model(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()

        self.norm = nn.BatchNorm2d(6)

        self.model = pretrainedmodels.resnet18(num_classes=1000, pretrained='imagenet')
        self.model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        embedding_size = self.model.last_linear.in_features
        self.model.last_linear = nn.Sequential()

        # self.embedding = nn.Embedding(4, embedding_size)

        self.output = nn.Sequential(
            nn.Dropout(model.dropout),
            nn.Linear(embedding_size, num_classes))

        # self.arc_output = nn.Sequential(
        #     nn.Dropout(model.dropout),
        #     NormalizedLinear(embedding_size, num_classes))
        # self.arc_face = ArcFace(num_classes)

    def forward(self, input, feats, target=None):
        if self.training:
            assert target is not None
        else:
            assert target is None

        input = self.norm(input)
        input = self.model(input)

        # embedding = self.embedding(feats)
        # input = input + embedding

        # if target is not None:
        #     alpha = torch.rand(target.size(0), 1).to(input.device)
        #     indices = get_shuffle_indices(target)
        #     input = alpha * input + (1 - alpha) * input[indices]

        output = self.output(input)
        # arc_output = self.arc_output(input)
        # arc_output = self.arc_face(arc_output, target)

        return output


def get_shuffle_indices(target):
    eq = target.unsqueeze(1) == target.unsqueeze(0)
    eq[torch.eye(target.size(0)).byte()] = 0
    indices = [np.where(row)[0] for row in eq.data.cpu().numpy()]
    indices = [np.random.choice(row) if row.shape[0] > 0 else i for i, row in enumerate(indices)]
    indices = torch.tensor(indices).to(target.device)

    return indices
