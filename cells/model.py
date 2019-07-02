import pretrainedmodels
import torch
import torch.nn as nn
import torch.nn.functional as F


# class GlobalGEMPool2d(nn.Module):
#     def __init__(self, p=2.):
#         super().__init__()
#
#         self.p = p
#
#     def forward(self, input):
#         input = input**self.p
#         input = F.adaptive_avg_pool2d(input, 1)
#         input = input**(1. / self.p)
#
#         return input

class GlobalGEMPool2d(nn.Module):
    def __init__(self):
        super().__init__()

        self.w = nn.Parameter(torch.tensor(0., dtype=torch.float))

    def forward(self, input):
        avg = F.adaptive_avg_pool2d(input, 1)
        max = F.adaptive_max_pool2d(input, 1)
        w = self.w.sigmoid()
        input = w * avg + (1. - w) * max
       
        return input


class Model(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()

        # self.model = pretrainedmodels.se_resnet50(num_classes=1000, pretrained='imagenet')
        # self.model.layer0.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.model.last_linear = nn.Linear(self.model.last_linear.in_features, num_classes)

        self.model = pretrainedmodels.resnet18(num_classes=1000, pretrained='imagenet')
        self.model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.model.avgpool = GlobalGEMPool2d()
        self.model.last_linear = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.last_linear.in_features, num_classes))

        self.norm = nn.BatchNorm2d(6)
        # self.drop = nn.Dropout2d(1 / 6)

    def forward(self, input):
        input = self.norm(input)
        # input = self.drop(input)
        input = self.model(input)

        return input
