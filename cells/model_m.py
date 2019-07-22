import efficientnet_pytorch
import torch.nn as nn

from cells.modules import ArcFace, NormalizedLinear


class Model(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()

        assert model.type in ['b0', 'b1', 'b2']

        self.norm = nn.BatchNorm2d(6)

        self.model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-{}'.format(model.type))
        # self.model._conv_stem = efficientnet_pytorch.utils.Conv2dDynamicSamePadding(
        #     6, 32, kernel_size=3, stride=2, bias=False)
        self.model._conv_stem = nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model._fc = NormalizedLinear(self.model._fc.in_features, num_classes)

        self.arc_face = ArcFace()

    def forward(self, input, feats, target=None):
        if self.training:
            assert target is not None
        else:
            assert target is None

        # assert input.size(2) == input.size(3) == self.model._global_params.image_size

        input = self.norm(input)
        input = self.model(input)

        output = input

        if target is None:
            arc_output = None
        else:
            arc_output = self.arc_face(input, target)

        return output, arc_output
