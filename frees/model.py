import torch
import pretrainedmodels
import librosa
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Model(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()

        self.spectrogram = Spectrogram(model.sample_rate)

        if model.type == 'max':
            self.model = MaxPoolModel(num_classes, dropout=model.dropout)
        # elif model.type == 'attn':
        #     self.model = AttentionModel(num_classes, dropout=model.dropout)
        else:
            raise AssertionError('invalid model {}'.format(model.type))

    def forward(self, input):
        images = self.spectrogram(input)
        logits, weights = self.model(images)

        return logits, images, weights


class MaxPoolModel(nn.Module):
    def __init__(self, num_classes, dropout):
        super().__init__()

        # block = torchvision.models.resnet.BasicBlock
        # self.model = torchvision.models.resnet.ResNet(block, [2, 2, 2, 2])
        # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.model.avgpool = nn.AdaptiveMaxPool2d(1)
        # self.model.fc = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(512 * block.expansion, num_classes))

        self.model = pretrainedmodels.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=3, bias=False)
        self.model.avgpool = nn.AdaptiveMaxPool2d(1)
        self.model.last_linear = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes))

    def forward(self, input):
        b, _, h, w = input.size()

        input = self.model(input)
        weights = torch.zeros(b, 1, h, w)

        return input, weights


# class AttentionModel(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#
#         block = torchvision.models.resnet.BasicBlock
#         self.model = torchvision.models.resnet.ResNet(block, [1, 1, 1, 1])
#         self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.model.fc = nn.Linear(512 * block.expansion, num_classes)
#         self.weights = nn.Conv1d(512 * block.expansion, 1, kernel_size=1)
#
#     def forward(self, input):
#         _, _, h, w = input.size()
#
#         input = self.model.conv1(input)
#         input = self.model.bn1(input)
#         input = self.model.relu(input)
#         input = self.model.maxpool(input)
#
#         input = self.model.layer1(input)
#         input = self.model.layer2(input)
#         input = self.model.layer3(input)
#         input = self.model.layer4(input)
#
#         input = input.mean(2)  # TODO: or max?
#         weights = self.weights(input)
#         weights = weights.softmax(2)
#         input = (input * weights).sum(2)
#
#         input = self.model.fc(input)
#
#         weights = weights.unsqueeze(2)
#         # weights = F.interpolate(weights,size=(h, w), mode='nearest')  # TODO: speedup, mode
#         weights = F.interpolate(weights, scale_factor=(4 * 4, 1 * 4), mode='nearest')  # TODO: speedup, mode
#
#         return input, weights


# class Swish(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, input):
#         return input * self.sigmoid(input)


# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
#         super().__init__()
#         self.conv1 = torchvision.models.resnet.conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.act = Swish()
#         self.conv2 = torchvision.models.resnet.conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         # self.se_module = senet.SEModule(planes, reduction=reduction)
#         self.se_module = NoOp()
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.act(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out = self.se_module(out) + identity
#         out = self.act(out)
#
#         return out
#
#
# class NoOp(nn.Module):
#     def forward(self, input):
#         return input


class Spectrogram(nn.Module):
    def __init__(self, rate):
        super().__init__()

        self.n_fft = round(0.025 * rate)
        self.hop_length = round(0.025 / 2 * rate)

        filters = librosa.filters.mel(rate, self.n_fft)
        filters = filters.reshape((*filters.shape, 1))
        filters = torch.tensor(filters).float()

        self.mel = nn.Conv1d(512, 128, 1, bias=False)
        self.mel.weight.data = filters
        self.mel.weight.requires_grad = False

        self.norm = nn.BatchNorm2d(1)

    def __call__(self, input):
        input = torch.stft(input, n_fft=self.n_fft, hop_length=self.hop_length)
        input = torch.norm(input, 2, -1)
        input = self.mel(input)
        input = torch.log(input + 1e-7)
        input = input.unsqueeze(1)
        input = self.norm(input)

        return input
