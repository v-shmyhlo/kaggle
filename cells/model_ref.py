import pretrainedmodels
import torch
import torch.nn as nn


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

        self.ref = nn.Sequential(
            nn.Linear(embedding_size * 18, embedding_size))

        self.output = nn.Sequential(
            nn.Dropout(model.dropout),
            nn.Linear(embedding_size, num_classes))

        # self.arc_output = nn.Sequential(
        #     nn.Dropout(model.dropout),
        #     NormalizedLinear(embedding_size, num_classes))
        # self.arc_face = ArcFace(num_classes)

    def forward(self, input, feats, target=None):
        ref = torch.zeros((18, 6, 224, 224)).to(input.device)

        if self.training:
            assert target is not None
        else:
            assert target is None

        input = torch.cat([input, ref], 0)
        input = self.norm(input)
        input = self.model(input)
        input, ref = input[:-ref.size(0)], input[-ref.size(0):]

        ref = ref.view(1, ref.size(0) * ref.size(1))
        ref = self.ref(ref)

        input = input - ref
        output = self.output(input)
       
        return output
