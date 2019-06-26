import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from detection.utils import boxes_yxhw_to_tlbr

STRIDES = [2**l for l in range(8)]


class ReLU(nn.ReLU):
    pass


class Norm(nn.GroupNorm):
    def __init__(self, num_features):
        super().__init__(num_channels=num_features, num_groups=32)


class ConvNorm(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2),
            Norm(out_channels))


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torchvision.models.resnet50(pretrained=True)

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

        # input = self.avgpool(input)
        # input = input.view(input.size(0), -1)
        # input = self.fc(input)

        input = [None, c1, c2, c3, c4, c5]

        return input


class UpsampleMerge(nn.Module):
    def __init__(self, c_channels):
        super().__init__()

        self.projection = ConvNorm(c_channels, 256, 1)
        self.output = ConvNorm(256, 256, 3)

    def forward(self, p, c):
        # TODO: assert sizes

        p = F.interpolate(p, size=(c.size(2), c.size(3)), mode='nearest')
        c = self.projection(c)
        input = p + c
        input = self.output(input)

        return input


# TODO: optimize level calculation
class FPN(nn.Module):
    def __init__(self, p2, p7):
        super().__init__()

        self.c5_to_p6 = ConvNorm(2048, 256, 3, stride=2)
        self.p6_to_p7 = nn.Sequential(
            ReLU(inplace=True),
            ConvNorm(256, 256, 3, stride=2)
        ) if p7 else None
        self.c5_to_p5 = ConvNorm(2048, 256, 1)
        self.p5c4_to_p4 = UpsampleMerge(1024)
        self.p4c3_to_p3 = UpsampleMerge(512)
        self.p3c2_to_p2 = UpsampleMerge(256) if p2 else None

    def forward(self, input):
        p6 = self.c5_to_p6(input[5])
        p7 = self.p6_to_p7(p6) if self.p6_to_p7 is not None else None
        p5 = self.c5_to_p5(input[5])
        p4 = self.p5c4_to_p4(p5, input[4])
        p3 = self.p4c3_to_p3(p4, input[3])
        p2 = self.p3c2_to_p2(p3, input[2]) if self.p3c2_to_p2 is not None else None

        input = [None, None, p2, p3, p4, p5, p6, p7]

        return input


class HeadSubnet(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            ConvNorm(in_channels, in_channels, 3),
            ReLU(inplace=True),
            ConvNorm(in_channels, in_channels, 3),
            ReLU(inplace=True),
            ConvNorm(in_channels, in_channels, 3),
            ReLU(inplace=True),
            ConvNorm(in_channels, in_channels, 3),
            ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1))


class FlattenDetectionMap(nn.Module):
    def __init__(self, num_anchors):
        super().__init__()

        self.num_anchors = num_anchors

    def forward(self, input):
        b, c, h, w = input.size()
        input = input.view(b, c // self.num_anchors, self.num_anchors * h * w)
        input = input.permute(0, 2, 1)

        return input


class MaskHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Sequential(
            ConvNorm(256, 256, 3),
            ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            ConvNorm(256, 128, 3),
            ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1))

    def forward(self, input):
        input = self.conv1(input)
        input = F.interpolate(input, scale_factor=2, mode='bilinear')
        input = self.conv2(input)

        return input


class RetinaNet(nn.Module):
    def __init__(self, num_classes, num_anchors):
        super().__init__()

        self.backbone = Backbone()
        self.fpn = FPN(p2=False, p7=True)
        self.class_head = HeadSubnet(256, num_anchors * num_classes)
        self.regr_head = HeadSubnet(256, num_anchors * 4)
        self.flatten = FlattenDetectionMap(num_anchors)

        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False

        modules = itertools.chain(
            self.fpn.modules(),
            self.class_head.modules(),
            self.regr_head.modules())
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0., std=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        pi = 0.01
        nn.init.constant_(self.class_head[-1].bias, -math.log((1 - pi) / pi))

    def forward(self, input):
        backbone_output = self.backbone(input)
        fpn_output = self.fpn(backbone_output)

        class_output = torch.cat([self.flatten(self.class_head(x)) for x in fpn_output if x is not None], 1)
        regr_output = torch.cat([self.flatten(self.regr_head(x)) for x in fpn_output if x is not None], 1)

        return class_output, regr_output

    def train(self, mode=True):
        super().train(mode)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class RetinaMaskNet(nn.Module):
    def __init__(self, num_classes, num_anchors):
        super().__init__()

        self.backbone = Backbone()
        self.fpn = FPN(p2=True, p7=True)
        self.class_head = HeadSubnet(256, num_anchors * num_classes)
        self.regr_head = HeadSubnet(256, num_anchors * 4)
        self.flatten = FlattenDetectionMap(num_anchors)
        self.mask_output = nn.Conv2d(256, num_classes + 1, 1)

        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False

        modules = itertools.chain(
            self.fpn.modules(),
            self.class_head.modules(),
            self.regr_head.modules())
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0., std=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        pi = 0.01
        nn.init.constant_(self.class_head[-1].bias, -math.log((1 - pi) / pi))

    def forward(self, input):
        backbone_output = self.backbone(input)
        fpn_output = self.fpn(backbone_output)

        mask_output = fpn_output[2]
        mask_output = self.mask_output(mask_output)
        mask_output = F.interpolate(mask_output, scale_factor=4, mode='bilinear')
        fpn_output = fpn_output[3:]

        class_output = torch.cat([self.flatten(self.class_head(x)) for x in fpn_output if x is not None], 1)
        regr_output = torch.cat([self.flatten(self.regr_head(x)) for x in fpn_output if x is not None], 1)

        return (class_output, regr_output), mask_output

    def train(self, mode=True):
        super().train(mode)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class RPN(nn.Module):
    def __init__(self, num_anchors):
        super().__init__()

        self.features = nn.Sequential(
            ConvNorm(256, 512, 3),
            ReLU(inplace=True))

        self.class_head = nn.Conv2d(512, num_anchors * 1, 1)
        self.regr_head = nn.Conv2d(512, num_anchors * 4, 1)

    def forward(self, input):
        input = self.features(input)

        class_output = self.class_head(input)
        regr_output = self.regr_head(input)

        return class_output, regr_output


# # TODO: clip boxes
# # TODO: assert box bounds
# class ROIAlign(nn.Module):
#     def __init__(self, size):
#         super().__init__()
#         self.size = size
#
#     def forward(self, fpn_output, boxes, image_ids):
#         level_ids = compute_level_ids(boxes)
#         boxes = boxes_yxhw_to_tlbr(boxes)
#
#         input = []
#         for b, i, l in zip(boxes, image_ids, level_ids):
#             b = (b / STRIDES[l]).long()
#
#             x = fpn_output[l][i:i + 1, :, b[0]:b[2], b[1]:b[3]]
#             x = F.interpolate(x, size=self.size, mode='bilinear')
#
#             input.append(x)
#
#         input = torch.cat(input, 0)
#
#         return input

# TODO: clip boxes
# TODO: assert box bounds
class ROIAlign(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.size = size

    def forward(self, fpn_output, boxes, image_ids):
        level_ids = compute_level_ids(boxes)
        boxes = boxes_yxhw_to_tlbr(boxes)

        input = []
        indices = torch.arange(boxes.size(0))
        order = []
        for level_id, fmap in enumerate(fpn_output):
            if fmap is None:
                continue

            level_boxes = []
            for image_id in range(fmap.size(0)):
                keep = (level_ids == level_id) & (image_ids == image_id)
                image_boxes = boxes[keep]
                image_boxes = image_boxes / STRIDES[level_id]
                level_boxes.append(image_boxes)
                order.append(indices[keep])

            x = torchvision.ops.roi_align(fmap, level_boxes, output_size=self.size)
            input.append(x)

        input = torch.cat(input, 0)
        order = torch.cat(order, 0).argsort()
        input = input[order]

        return input


class ROIHead(nn.Module):
    def __init__(self, size, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(256, 512, size),
            Norm(512),
            ReLU(inplace=True))

        self.class_head = nn.Conv2d(512, num_classes, 1)
        self.regr_head = nn.Conv2d(512, 4, 1)

    def forward(self, input):
        input = self.conv(input)
        class_output = self.class_head(input)
        regr_output = self.regr_head(input)

        class_output = class_output.view(class_output.size(0), class_output.size(1))
        regr_output = regr_output.view(regr_output.size(0), regr_output.size(1))

        return class_output, regr_output


# TODO: parameterize fpn
class MaskRCNN(nn.Module):
    def __init__(self, num_classes, num_anchors):
        super().__init__()

        self.backbone = Backbone()
        self.fpn = FPN(p2=True, p7=False)
        self.rpn = RPN(num_anchors)
        self.flatten = FlattenDetectionMap(num_anchors)
        self.roi_align = ROIAlign(7)
        self.roi_head = ROIHead(7, num_classes)

        modules = itertools.chain(
            self.fpn.modules(),
            self.rpn.modules())
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0., std=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        backbone_output = self.backbone(input)
        fpn_output = self.fpn(backbone_output)

        proposals = [self.rpn(x) for x in fpn_output if x is not None]
        class_output, regr_output = zip(*proposals)

        class_output = torch.cat([self.flatten(x) for x in class_output], 1)
        regr_output = torch.cat([self.flatten(x) for x in regr_output], 1)

        return fpn_output, (class_output, regr_output)

    def roi(self, fpn_output, boxes, image_ids):
        input = self.roi_align(fpn_output, boxes, image_ids)
        input = self.roi_head(input)

        return input


def compute_level_ids(boxes):
    k0 = 4
    # k_min = 2
    # k_max = 5
    k_min = 3
    k_max = 6

    areas = boxes[:, 2] * boxes[:, 3]

    k = torch.floor(k0 + torch.log2(areas.sqrt() / 224))
    k = torch.clamp(k, min=k_min, max=k_max)
    k = k.long()

    return k
