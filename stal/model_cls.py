import itertools

# import efficientnet_pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# TODO: init

class ReLU(nn.ReLU):
    pass


class Norm(nn.BatchNorm2d):
    pass


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=True):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, groups=groups, bias=bias)


class ConvNorm(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super().__init__(
            Conv(in_channels, out_channels, kernel_size, stride=stride, groups=groups, bias=False),
            Norm(out_channels))


class UpsampleMerge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.bottom = nn.Sequential(
            ConvNorm(in_channels, out_channels, 1),
            ReLU(inplace=True))

        # self.refine = nn.Sequential(
        #     ConvNorm(out_channels, out_channels, 3),
        #     ReLU(inplace=True))

    def forward(self, bottom, left):
        bottom = self.bottom(bottom)
        bottom = F.interpolate(bottom, scale_factor=2, mode='bilinear')
        input = bottom + left
        # input = self.refine(input)

        return input


class ResNetEncoder(nn.Module):
    sizes = [None, 64, 64, 128, 256, 512]

    def __init__(self, pretrained):
        super().__init__()

        self.model = torchvision.models.resnet18(pretrained=pretrained)

    def forward(self, input):
        input = self.model.conv1(input)
        input = self.model.bn1(input)
        input = self.model.relu(input)
        c1 = input
        input = self.model.maxpool(input)
        input = self.model.layer1(input)
        c2 = input
        input = self.model.layer2(input)
        c3 = input
        input = self.model.layer3(input)
        c4 = input
        input = self.model.layer4(input)
        c5 = input

        return [None, c1, c2, c3, c4, c5]


class EffNetEncoder(nn.Module):
    sizes = [None, 16, 24, 40, 112, 320]

    def __init__(self, pretrained):
        super().__init__()

        self.model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = efficientnet_pytorch.model.relu_fn(self.model._bn0(self.model._conv_stem(inputs)))

        # Blocks
        blocks = []
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            blocks.append(x)

        blocks = itertools.groupby(blocks, key=lambda x: x.size()[2:])
        fmaps = [None]
        for _, stage in blocks:
            fmap = list(stage)[-1]
            fmaps.append(fmap)
        assert len(fmaps) == 6

        # Head
        # x = efficientnet_pytorch.model.relu_fn(self.model._bn1(self.model._conv_head(x)))

        return fmaps

    def forward(self, input):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """

        # Convolution layers
        input = self.extract_features(input)

        # # Pooling and final linear layer
        # input = F.adaptive_avg_pool2d(input, 1).squeeze(-1).squeeze(-1)
        # if self._dropout:
        #     input = F.dropout(input, p=self._dropout, training=self.training)
        # input = self._fc(input)

        return input


class Decoder(nn.Module):
    def __init__(self, sizes):
        super().__init__()

        self.merge1 = UpsampleMerge(sizes[2], sizes[1])
        self.merge2 = UpsampleMerge(sizes[3], sizes[2])
        self.merge3 = UpsampleMerge(sizes[4], sizes[3])
        self.merge4 = UpsampleMerge(sizes[5], sizes[4])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, fmaps):
        input = fmaps[5]

        input = self.merge4(input, fmaps[4])
        input = self.merge3(input, fmaps[3])
        input = self.merge2(input, fmaps[2])
        input = self.merge1(input, fmaps[1])

        return input


class Model(nn.Module):
    def __init__(self, model, num_classes, pretrained=True):
        super().__init__()

        self.norm = nn.BatchNorm2d(3)
        if model.encoder == 'resnet':
            self.encoder = ResNetEncoder(pretrained=pretrained)
        elif model.encoder == 'effnet':
            self.encoder = EffNetEncoder(pretrained=pretrained)
        else:
            raise AssertionError('invalid model.encoder {}'.format(model.encoder))
        self.decoder = Decoder(self.encoder.sizes)
        self.output = Conv(self.encoder.sizes[1], num_classes, 1)

        self.pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(self.encoder.sizes[5], num_classes)

    def forward(self, input):
        input = self.norm(input)
        input = self.encoder(input)

        classifier = input[5]
        classifier = self.pool(classifier)
        classifier = classifier.view(classifier.size(0), classifier.size(1))
        classifier = self.classifier(classifier)

        input = self.decoder(input)
        input = self.output(input)
        input = F.interpolate(input, scale_factor=2, mode='bilinear')

        return classifier, input


class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()

        self.models = nn.ModuleList(models)

    def forward(self, input):
        outputs = [model(input) for model in self.models]

        class_logits, mask_logits = zip(*outputs)

        class_logits = torch.stack(class_logits, 1)
        mask_logits = torch.stack(mask_logits, 1)

        return class_logits, mask_logits
