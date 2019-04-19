import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import pretrainedmodels
import math
from PIL import Image
from collections import OrderedDict
import argparse
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold
from scheduler import OneCycleScheduler
from utils import Mean, seed_everything
from augmentation import SquarePad

# TODO: check del

FOLDS = list(range(1, 5 + 1))

parser = argparse.ArgumentParser()
parser.add_argument('--experiment-path', type=str, default='./tf_log/imet')
parser.add_argument('--dataset-path', type=str, required=True)
parser.add_argument('--workers', type=int, default=os.cpu_count())
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--image-size', type=int, default=128)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--annealing', type=str, choices=['linear', 'cosine'], default='linear')
parser.add_argument('--aug', type=str, choices=['low', 'med', 'med+color', 'hard', 'pad'], default='med')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

seed_everything(args.seed)

train_data = pd.read_csv(os.path.join(args.dataset_path, 'train.csv'))
train_data['attribute_ids'] = train_data['attribute_ids'].apply(lambda s: [int(x) for x in s.split()])

classes = pd.read_csv(os.path.join(args.dataset_path, 'labels.csv'))
id_to_class = {row['attribute_id']: row['attribute_name'] for _, row in classes.iterrows()}
class_to_id = {id_to_class[k]: k for k in id_to_class}


class TrainEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data.iloc[i]

        image = load_image(os.path.join(args.dataset_path, 'train/{}.png'.format(row['id'])))
        if self.transform is not None:
            image = self.transform(image)

        label = np.zeros(NUM_CLASSES, dtype=np.float32)
        for l in row['attribute_ids']:  # TODO:
            label[l] = 1.

        return image, label, row['id']


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.data = os.listdir(os.path.join(args.dataset_path, 'test'))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path = self.data[i]
        id = os.path.splitext(path)[0]

        image = load_image(os.path.join(args.dataset_path, 'test', path))
        if self.transform is not None:
            image = self.transform(image)

        return image, id


# TODO: add images

# def load_image(path, size):
#     image = cv2.imread(path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, size)

#     return image

# def load_image(path, size):
#     image = Image.open(path)
#     image = image.resize(size)
#     image = np.array(image)

#     return image

def load_image(path):
    if args.debug:
        path = './imet/dog.jpg'

    image = Image.open(path)

    return image


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        if ARCH == 'resnet34':
            block = torchvision.models.resnet.BasicBlock
            self.model = ResNet(block, [3, 4, 6, 3])
            self.model.load_state_dict(torch.utils.model_zoo.load_url(torchvision.models.resnet.model_urls['resnet34']))
            self.model.fc = nn.Linear(512 * block.expansion, NUM_CLASSES)
        elif ARCH == 'resnet50':
            block = torchvision.models.resnet.Bottleneck
            self.model = ResNet(block, [3, 4, 6, 3])
            self.model.load_state_dict(torch.utils.model_zoo.load_url(torchvision.models.resnet.model_urls['resnet50']))
            self.model.fc = nn.Linear(512 * block.expansion, NUM_CLASSES)
        elif ARCH == 'seresnext50':
            block = SEResNeXtBottleneck
            self.model = SENet(
                block, [3, 4, 6, 3], groups=32, reduction=16, dropout_p=None,
                inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0,
                num_classes=1000)
            settings = pretrainedmodels.models.senet.pretrained_settings['se_resnext50_32x4d']['imagenet']
            pretrainedmodels.models.senet.initialize_pretrained_model(self.model, 1000, settings)
            self.model.last_linear = nn.Linear(512 * block.expansion, NUM_CLASSES)
        else:
            raise AssertionError('invalid ARCH {}'.format(ARCH))

    def forward(self, input):
        return self.model(input)


class Attention(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.weight = nn.Conv2d(in_features, 1, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, input):
        b, c, h, w = input.size()

        weight = self.weight(input)
        weight = weight.view(b, 1, h * w)
        weight = weight.softmax(-1)

        input = input.view(b, c, h * w)
        input = input * weight
        input = input.sum(-1)
        input = input.view(b, c, 1, 1)

        return input


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class SENet(nn.Module):
    def __init__(self, block, layers, groups, reduction, dropout_p=0.2, inplanes=128, input_3x3=True,
                 downsample_kernel_size=3, downsample_padding=1, num_classes=1000):
        super(SENet, self).__init__()

        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0)
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion, kernel_size=downsample_kernel_size,
                    stride=stride, padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)

        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)

        return x


class Bottleneck(nn.Module):
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEResNeXtBottleneck(Bottleneck):
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()

        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x

        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class EWA(object):
    def __init__(self, beta=0.9):
        self.beta = beta
        self.step = 0
        self.average = 0

    def update(self, value):
        self.step += 1
        self.average = self.beta * self.average + (1 - self.beta) * value

    def compute(self):
        return self.average / (1 - self.beta**self.step)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        target = target.float()
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size()) == 2:
            loss = loss.sum(dim=1)
        return loss.mean()


def f2_loss(input, target, eps=1e-7):
    input = input.sigmoid()

    tp = (target * input).sum(1)
    tn = ((1 - target) * (1 - input)).sum(1)
    fp = ((1 - target) * input).sum(1)
    fn = (target * (1 - input)).sum(1)

    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)

    beta_sq = 2**2
    f2 = (1 + beta_sq) * p * r / (beta_sq * p + r + eps)
    loss = -(f2 + eps).log()

    return loss


def hinge_loss(input, target, delta=1.):
    target = target * 1. + (1 - target) * -1.
    loss = torch.max(torch.zeros_like(input).to(input.device), delta - target * input)

    return loss


def compute_loss(input, target):
    loss = [l(input=input, target=target).mean() for l in LOSS]
    loss = sum(loss) / len(loss)

    return loss


def compute_score(input, target, threshold=0.5):
    input = (input.sigmoid() > threshold).float()

    tp = (target * input).sum(-1)
    tn = ((1 - target) * (1 - input)).sum(-1)
    fp = ((1 - target) * input).sum(-1)
    fn = (target * (1 - input)).sum(-1)

    p = tp / (tp + fp)
    r = tp / (tp + fn)

    beta_sq = 2**2
    f2 = (1 + beta_sq) * p * r / (beta_sq * p + r)
    f2[f2 != f2] = 0.

    return f2


def find_threshold_global(input, target):
    thresholds = np.arange(0.1, 0.9, 0.01)
    scores = [compute_score(input=input, target=target, threshold=t).mean()
              for t in tqdm(thresholds, desc='threshold search')]
    threshold = thresholds[np.argmax(scores)]
    score = scores[np.argmax(scores)]

    # plt.plot(thresholds, scores)
    # plt.axvline(threshold)
    # plt.title('score: {:.4f}, threshold: {:.4f}'.format(score.item(), threshold))
    # plt.show()

    return threshold, score


def find_threshold_class(input, target, initial):
    threshold = torch.full((NUM_CLASSES,), initial).to(input.device)
    steps = []
    for _ in tqdm(range(50), desc='threshold search'):
        for i in np.random.permutation(NUM_CLASSES):
            r = torch.tensor([threshold[i] - 0.01, threshold[i], threshold[i] + 0.01])
            space = threshold.view(1, NUM_CLASSES).repeat(r.size(0), 1)
            space[:, i] = r
            scores = compute_score(input=input.unsqueeze(0), target=target.unsqueeze(0), threshold=space.unsqueeze(1))
            scores = scores.mean(-1)
            threshold[i] = r[scores.argmax()]

        score = compute_score(input=input, target=target, threshold=threshold).mean()
        steps.append(score)

    # plt.plot(steps)
    # plt.show()

    # plt.hist(threshold.cpu(), bins=50)
    # plt.title('score: {:.4f}, threshold mean: {:.4f}, threshold std: {:.4f}'.format(
    #     score.item(), threshold.mean(), threshold.std()))
    # plt.show()

    return threshold


NUM_CLASSES = len(classes)
ARCH = 'seresnext50'
OPT = 'adam'
LOSS_SMOOTHING = 0.9
LOSS = [
    # f2_loss,
    # F.binary_cross_entropy_with_logits,
    FocalLoss(),
    # hinge_loss,
]
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# TODO: pin memory
# TODO: stochastic weight averaging
# TODO: group images by buckets (size, ratio) and batch
# TODO: hinge loss clamp instead of minimum
# TODO: losses
# TODO: better one cycle
# TODO: cos vs lin
# TODO: load and restore state after lr finder
# TODO: better loss smoothing
# TODO: shuffle thresh search
# TODO: init thresh search from global best
# TODO: shuffle split
# TODO: tune on large size
# TODO: cross val
# TODO: smart sampling
# TODO: larger model
# TODO: imagenet papers
# TODO: load image as jpeg
# TODO: augmentations (flip, crops, color)
# TODO: min 1 tag?
# TODO: pick threshold to match ratio
# TODO: compute smoothing beta from batch size and num steps
# TODO: speedup image loading
# TODO: pin memory
# TODO: smart sampling
# TODO: better threshold search (step, epochs)
# TODO: weight standartization
# TODO: label smoothing
# TODO: build sched for lr find


to_tensor_and_norm = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if args.aug == 'low':
    train_transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.RandomHorizontalFlip(),
        to_tensor_and_norm,
    ])
    eval_transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        to_tensor_and_norm,
    ])
elif args.aug == 'med':
    train_transform = T.Compose([
        T.RandomResizedCrop((args.image_size, args.image_size), scale=(1., 1.), ratio=(3. / 4., 4. / 3.)),
        T.RandomHorizontalFlip(),
        to_tensor_and_norm,
    ])
    eval_transform = T.Compose([
        T.Resize(args.image_size),
        T.CenterCrop(args.image_size),
        to_tensor_and_norm,
    ])
elif args.aug == 'med+color':
    train_transform = T.Compose([
        T.RandomResizedCrop((args.image_size, args.image_size), scale=(1., 1.), ratio=(3. / 4., 4. / 3.)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        to_tensor_and_norm,
    ])
    eval_transform = T.Compose([
        T.Resize(args.image_size),
        T.CenterCrop(args.image_size),
        to_tensor_and_norm,
    ])
elif args.aug == 'hard':
    scale = (0.6, 1.0)
    # image_size_corrected = round(args.image_size * (1 / np.mean(scale).item()))
    image_size_corrected = args.image_size

    train_transform = T.Compose([
        T.RandomResizedCrop((args.image_size, args.image_size), scale=scale, ratio=(3. / 4., 4. / 3.)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        to_tensor_and_norm,
    ])
    # TODO: correct eval size?
    eval_transform = T.Compose([
        T.Resize(image_size_corrected),
        T.CenterCrop(image_size_corrected),
        to_tensor_and_norm,
    ])
elif args.aug == 'pad':
    image_size_corrected = round(args.image_size * (1 / 0.8))

    train_transform = T.Compose([
        SquarePad(padding_mode='edge'),
        T.Resize(image_size_corrected),
        T.RandomCrop(args.image_size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        to_tensor_and_norm,
    ])
    eval_transform = T.Compose([
        SquarePad(padding_mode='edge'),
        T.Resize(image_size_corrected),
        T.CenterCrop(args.image_size),
        to_tensor_and_norm,
    ])
else:
    raise AssertionError('invalid AUG {}'.format(args.aug))


def build_optimizer(parameters, lr, weight_decay):
    if OPT == 'adam':
        return torch.optim.Adam(parameters, lr, weight_decay=weight_decay)
    elif OPT == 'sgd':
        return torch.optim.SGD(parameters, lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise AssertionError('invalid OPT {}'.format(OPT))


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def find_lr():
    train_dataset = TrainEvalDataset(train_data, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True,
        num_workers=args.workers)  # TODO: all args

    min_lr = 1e-8
    max_lr = 10.
    gamma = (max_lr / min_lr)**(1 / len(train_data_loader))

    stp = 0
    cur_lr = min_lr
    lrs = []
    lss = []
    sls = []
    ewa = EWA(beta=LOSS_SMOOTHING)

    minima = {
        'loss': np.inf,
        'lr': min_lr
    }

    model = Model()
    model = model.to(DEVICE)
    optimizer = build_optimizer(model.parameters(), min_lr, weight_decay=args.weight_decay)

    writer = SummaryWriter(os.path.join(args.experiment_path, 'lr_search'))

    model.train()
    for images, labels, ids in tqdm(train_data_loader, desc='lr search'):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)

        loss = compute_loss(input=logits, target=labels)

        ewa.update(loss.data.cpu().numpy().mean())
        lrs.append(cur_lr)
        lss.append(loss.data.cpu().numpy().mean())
        sls.append(ewa.compute())
        if ewa.compute() < minima['loss']:
            minima['loss'] = ewa.compute()
            minima['lr'] = cur_lr
        if minima['loss'] * 4 < ewa.compute():
            break
        stp += 1

        cur_lr = min_lr * gamma**stp
        set_lr(optimizer, cur_lr)

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        writer.add_scalar('loss', loss.mean().data.cpu().numpy(), global_step=stp)

        if args.debug:
            break

    return minima


def indices_for_fold(fold, size):
    kfold = KFold(len(FOLDS), shuffle=True, random_state=args.seed)
    splits = list(kfold.split(np.zeros(size)))
    train_indices, eval_indices = splits[fold - 1]
    assert len(train_indices) + len(eval_indices) == size

    return train_indices, eval_indices


def train_fold(fold, minima):
    train_indices, eval_indices = indices_for_fold(fold, len(train_data))

    train_dataset = TrainEvalDataset(train_data.iloc[train_indices], transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True,
        num_workers=args.workers)  # TODO: all args

    eval_dataset = TrainEvalDataset(train_data.iloc[eval_indices], transform=eval_transform)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, num_workers=args.workers)  # TODO: all args

    model = Model()
    model = model.to(DEVICE)
    optimizer = build_optimizer(model.parameters(), 0., weight_decay=args.weight_decay)
    scheduler = OneCycleScheduler(
        optimizer, lr=(minima['lr'] / 10 / 25, minima['lr'] / 10), beta=(0.95, 0.85),
        max_steps=len(train_data_loader) * args.epochs, annealing=args.annealing)

    metrics = {
        'loss': Mean(),
    }
    train_writer = SummaryWriter(os.path.join(args.experiment_path, 'train', 'fold{}'.format(fold)))
    eval_writer = SummaryWriter(os.path.join(args.experiment_path, 'eval', 'fold{}'.format(fold)))

    best_score = 0
    for epoch in range(args.epochs):
        model.train()
        for images, labels, ids in tqdm(train_data_loader, desc='epoch {} train'.format(epoch)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)

            loss = compute_loss(input=logits, target=labels)
            metrics['loss'].update(loss.data.cpu().numpy())

            scheduler.step()
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            if args.debug:
                break

        loss = metrics['loss'].compute_and_reset()
        print('[FOLD {}][EPOCH {}][TRAIN] loss: {:.4f}'.format(fold, epoch, loss))
        train_writer.add_scalar('loss', loss, global_step=epoch)
        lr, beta = scheduler.get_lr()
        train_writer.add_scalar('lr', lr, global_step=epoch)
        train_writer.add_scalar('beta', beta, global_step=epoch)
        train_writer.add_image('image', torchvision.utils.make_grid(images[:32], normalize=True), global_step=epoch)

        predictions = []
        targets = []
        model.eval()
        with torch.no_grad():
            for images, labels, ids in tqdm(eval_data_loader, desc='epoch {} evaluation'.format(epoch)):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits = model(images)

                targets.append(labels)
                predictions.append(logits)

                loss = compute_loss(input=logits, target=labels)
                metrics['loss'].update(loss.data.cpu().numpy())

                if args.debug:
                    break

            loss = metrics['loss'].compute_and_reset()

            predictions = torch.cat(predictions, 0)
            targets = torch.cat(targets, 0)
            threshold, score = find_threshold_global(input=predictions, target=targets)

            print('[FOLD {}][EPOCH {}][EVAL] loss: {:.4f}, score: {:.4f}'.format(fold, epoch, loss, score))
            eval_writer.add_scalar('loss', loss, global_step=epoch)
            eval_writer.add_scalar('score', score, global_step=epoch)

            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), './model_{}.pth'.format(fold))


def build_submission(threshold):
    with torch.no_grad():
        predictions = 0.

        for fold in FOLDS:
            fold_predictions, fold_ids = predict_on_test_using_fold(fold)
            predictions = predictions + fold_predictions.sigmoid()
            ids = fold_ids

        predictions = predictions / len(FOLDS)
        submission = []
        assert len(ids) == len(predictions)
        for id, prediction in zip(ids, predictions):
            pred = (prediction > threshold).nonzero().reshape(-1)
            pred = pred.data.cpu().numpy()
            pred = map(str, pred)
            pred = ' '.join(pred)

            submission.append((id, pred))

        submission = pd.DataFrame(submission, columns=['id', 'attribute_ids'])
        submission.to_csv('./submission.csv', index=False)


def predict_on_test_using_fold(fold):
    test_dataset = TestDataset(transform=eval_transform)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.workers)  # TODO: all args

    model = Model()
    model = model.to(DEVICE)
    model.load_state_dict(torch.load('./model_{}.pth'.format(fold)))
    model.eval()

    with torch.no_grad():
        fold_predictions = []
        fold_ids = []
        for images, ids in tqdm(test_data_loader, desc='fold {} inference'.format(fold)):
            images = images.to(DEVICE)
            logits = model(images)
            fold_predictions.append(logits)
            fold_ids.extend(ids)

            if args.debug:
                break

        fold_predictions = torch.cat(fold_predictions, 0)

    return fold_predictions, fold_ids


def predict_on_eval_using_fold(fold):
    _, eval_indices = indices_for_fold(fold, len(train_data))

    eval_dataset = TrainEvalDataset(train_data.iloc[eval_indices], transform=eval_transform)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, num_workers=args.workers)  # TODO: all args

    model = Model()
    model = model.to(DEVICE)
    model.load_state_dict(torch.load('./model_{}.pth'.format(fold)))
    model.eval()

    with torch.no_grad():
        fold_targets = []
        fold_predictions = []
        fold_ids = []
        for images, labels, ids in tqdm(eval_data_loader, desc='fold {} best model evaluation'.format(fold)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)

            fold_targets.append(labels)
            fold_predictions.append(logits)
            fold_ids.extend(ids)

            if args.debug:
                break

        fold_targets = torch.cat(fold_targets, 0)
        fold_predictions = torch.cat(fold_predictions, 0)

    return fold_targets, fold_predictions, fold_ids


def find_threshold_for_folds():
    with torch.no_grad():
        targets = []
        predictions = []
        for fold in FOLDS:
            fold_targets, fold_predictions, fold_ids = predict_on_eval_using_fold(fold)
            targets.append(fold_targets)
            predictions.append(fold_predictions)

        predictions = torch.cat(predictions, 0)
        targets = torch.cat(targets, 0)
        threshold, score = find_threshold_global(input=predictions, target=targets)

        print('threshold: {:.4f}, score: {:.4f}'.format(threshold, score))

        return threshold


def main():
    minima = find_lr()
    for fold in FOLDS:
        train_fold(fold, minima)
    threshold = find_threshold_for_folds()
    build_submission(threshold)


if __name__ == '__main__':
    main()
