import torch.autograd as ag
import torch.nn as nn
import torchvision


class GradientReversal(ag.Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.alpha = alpha

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class Model(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()

        # # self.model = pretrainedmodels.se_resnet50(num_classes=1000, pretrained='imagenet')
        # # self.model.layer0.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # # self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        # # self.model.last_linear = nn.Linear(self.model.last_linear.in_features, num_classes)
        #
        # self.model = pretrainedmodels.resnet18(num_classes=1000, pretrained='imagenet')
        # self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        # # self.model.avgpool = GlobalGEMPool2d(512)
        #
        # # self.drop = nn.Dropout2d(1 / 6)

        self.norm = nn.BatchNorm2d(6)
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes))
        self.domain = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1))

    def forward(self, input, alpha=1.):
        input = self.norm(input)

        input = self.model.conv1(input)
        input = self.model.bn1(input)
        input = self.model.relu(input)
        input = self.model.maxpool(input)

        input = self.model.layer1(input)
        input = self.model.layer2(input)
        input = self.model.layer3(input)
        input = self.model.layer4(input)

        input = self.model.avgpool(input)
        input = input.reshape(input.size(0), -1)

        output = self.model.fc(input)
        domain = self.domain(GradientReversal.apply(input, alpha)).squeeze(1)

        return output, domain
