import random
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import KFold
from collections import OrderedDict

print(os.listdir('../input'))

FOLDS = list(range(1, 5 + 1))
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class Args(object):
    dataset_path = '../input/imet-2019-fgvc6'
    experiment_path = '../input/imetmodel'
    workers = 0


class Config(object):
    class Aug(object):
        type = 'rpad'
        crop_scale = 0.875  # 224 / 256

    class Model(object):
        type = 'seresnext50'
        predict_thresh = True
        dropout = 0.2

    seed = 42
    image_size = 320
    batch_size = 38
    aug = Aug()
    model = Model()


args = Args()
config = Config()

train_data = pd.read_csv(os.path.join(args.dataset_path, 'train.csv'))
train_data['attribute_ids'] = train_data['attribute_ids'].apply(lambda s: [int(x) for x in s.split()])

classes = pd.read_csv(os.path.join(args.dataset_path, 'labels.csv'))


class Model(nn.Module):
    def __init__(self, arch, num_classes):
        super().__init__()

        if arch.predict_thresh:
            num_classes *= 2

        if arch.type == 'seresnext50':
            block = SEResNeXtBottleneck
            self.model = SENet(
                block,
                [3, 4, 6, 3],
                groups=32,
                reduction=16,
                dropout_p=arch.dropout,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                num_classes=1000)
            # settings = pretrainedmodels.models.senet.pretrained_settings['se_resnext50_32x4d']['imagenet']
            # pretrainedmodels.models.senet.initialize_pretrained_model(self.model, 1000, settings)
            self.model.last_linear = nn.Linear(512 * block.expansion, num_classes)
        else:
            raise AssertionError('invalid ARCH {}'.format(arch.type))

    def forward(self, input):
        input = self.model(input)

        return input


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


class SquarePad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        max_side = np.max(img.size)
        left = (max_side - img.size[0]) // 2
        right = max_side - img.size[0] - left
        top = (max_side - img.size[1]) // 2
        bottom = max_side - img.size[1] - top

        return TF.pad(img, (left, top, right, bottom), self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(fill={0}, padding_mode={1})'.format(self.fill, self.padding_mode)


class RatioPad(object):
    def __init__(self, ratio=(2 / 3, 3 / 2), fill=0, padding_mode='constant'):
        assert ratio[0] < ratio[1], '{} should be less than {}'.format(ratio[0], ratio[1])

        self.ratio = ratio
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        w, h = img.size
        ratio = w / h

        if ratio < self.ratio[0]:
            w = round(h * self.ratio[0])
        elif ratio > self.ratio[1]:
            h = round(w / self.ratio[1])

        left = (w - img.size[0]) // 2
        right = w - img.size[0] - left
        top = (h - img.size[1]) // 2
        bottom = h - img.size[1] - top

        return TF.pad(img, (left, top, right, bottom), self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(ratio={0}, fill={1}, padding_mode={2})'.format(
            self.ratio, self.fill, self.padding_mode)


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        _, h, w = img.size()

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(-self.length // 2, h + self.length // 2)
            x = np.random.randint(-self.length // 2, w + self.length // 2)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


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
        label[row['attribute_ids']] = 1.

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


def load_image(path):
    image = Image.open(path)

    return image


def worker_init_fn(_):
    seed_python(torch.initial_seed() % 2**32)


def output_to_logits(input):
    if config.model.predict_thresh:
        logits, thresholds = input.split(input.size(-1) // 2, -1)
        logits = logits - thresholds
    else:
        logits = input

    return logits


def compute_score(input, target, threshold=0.5):
    input = output_to_logits(input)
    input = (input.sigmoid() > threshold).float()

    tp = (target * input).sum(-1)
    # tn = ((1 - target) * (1 - input)).sum(-1)
    fp = ((1 - target) * input).sum(-1)
    fn = (target * (1 - input)).sum(-1)

    p = tp / (tp + fp)
    r = tp / (tp + fn)

    beta_sq = 2**2
    f2 = (1 + beta_sq) * p * r / (beta_sq * p + r)
    f2[f2 != f2] = 0.

    return f2


def find_threshold_global(input, target):
    thresholds = np.arange(0.01, 1 - 0.01, 0.01)
    scores = [compute_score(input=input, target=target, threshold=t).mean()
              for t in tqdm(thresholds, desc='threshold search')]
    threshold = thresholds[np.argmax(scores)]
    score = scores[np.argmax(scores)]

    fig = plt.figure()
    plt.plot(thresholds, scores)
    plt.axvline(threshold)
    plt.title('score: {:.4f}, threshold: {:.4f}'.format(score.item(), threshold))

    return threshold, score, fig


NUM_CLASSES = len(classes)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

image_size_corrected = round(config.image_size * (1 / config.aug.crop_scale))

if config.aug.type == 'resize':
    resize = T.Resize((image_size_corrected, image_size_corrected))

    eval_transform = resize
    test_transform = resize
elif config.aug.type == 'crop':
    resize = T.Resize(image_size_corrected)

    eval_transform = resize
    test_transform = resize
elif config.aug.type == 'pad':
    pad_and_resize = T.Compose([
        SquarePad(padding_mode='edge'),
        T.Resize(image_size_corrected),
    ])

    eval_transform = pad_and_resize
    test_transform = pad_and_resize
elif config.aug.type == 'rpad':
    pad_and_resize = T.Compose([
        RatioPad(padding_mode='edge'),
        T.Resize(image_size_corrected),
    ])

    eval_transform = pad_and_resize
    test_transform = pad_and_resize
else:
    raise AssertionError('invalid aug {}'.format(config.aug.type))

to_tensor_and_norm = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])
eval_transform = T.Compose([
    eval_transform,
    T.CenterCrop(config.image_size),
    to_tensor_and_norm,
])
test_transform = T.Compose([
    test_transform,
    T.TenCrop(config.image_size),
    T.Lambda(lambda xs: torch.stack([to_tensor_and_norm(x) for x in xs], 0))
])


def indices_for_fold(fold, dataset_size):
    kfold = KFold(len(FOLDS), shuffle=True, random_state=config.seed)
    splits = list(kfold.split(np.zeros(dataset_size)))
    train_indices, eval_indices = splits[fold - 1]
    assert len(train_indices) + len(eval_indices) == dataset_size

    return train_indices, eval_indices


def build_submission(folds, threshold):
    with torch.no_grad():
        predictions = 0.

        for fold in folds:
            fold_predictions, fold_ids = predict_on_test_using_fold(fold)
            fold_predictions = output_to_logits(fold_predictions)
            fold_predictions = fold_predictions.sigmoid().mean(1)

            predictions = predictions + fold_predictions
            ids = fold_ids

        predictions = predictions / len(folds)
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
    test_dataset = TestDataset(transform=test_transform)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size // 2,
        num_workers=args.workers,
        worker_init_fn=worker_init_fn)

    model = Model(config.model, NUM_CLASSES)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(args.experiment_path, 'model_{}.pth'.format(fold))))

    model.eval()
    with torch.no_grad():
        fold_predictions = []
        fold_ids = []
        for images, ids in tqdm(test_data_loader, desc='fold {} inference'.format(fold)):
            images = images.to(DEVICE)

            b, n, c, h, w = images.size()
            images = images.view(b * n, c, h, w)
            logits = model(images)
            logits = logits.view(b, n, NUM_CLASSES * (1 + config.model.predict_thresh))

            fold_predictions.append(logits)
            fold_ids.extend(ids)

        fold_predictions = torch.cat(fold_predictions, 0)

    return fold_predictions, fold_ids


def predict_on_eval_using_fold(fold):
    _, eval_indices = indices_for_fold(fold, len(train_data))

    eval_dataset = TrainEvalDataset(train_data.iloc[eval_indices], transform=eval_transform)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        num_workers=args.workers,
        worker_init_fn=worker_init_fn)

    model = Model(config.model, NUM_CLASSES)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(args.experiment_path, 'model_{}.pth'.format(fold))))

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

        fold_targets = torch.cat(fold_targets, 0)
        fold_predictions = torch.cat(fold_predictions, 0)

    return fold_targets, fold_predictions, fold_ids


def find_threshold_for_folds(folds):
    with torch.no_grad():
        targets = []
        predictions = []
        for fold in folds:
            fold_targets, fold_predictions, fold_ids = predict_on_eval_using_fold(fold)
            targets.append(fold_targets)
            predictions.append(fold_predictions)

        # TODO: check aggregated correctly
        predictions = torch.cat(predictions, 0)
        targets = torch.cat(targets, 0)
        threshold, score, _ = find_threshold_global(input=predictions, target=targets)

        print('threshold: {:.4f}, score: {:.4f}'.format(threshold, score))

        return threshold


def seed_python(seed):
    random.seed(seed)
    np.random.seed(seed)


def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    seed_python(config.seed)
    seed_torch(config.seed)

    folds = FOLDS
    threshold = find_threshold_for_folds(folds)
    build_submission(folds, threshold)


if __name__ == '__main__':
    main()
