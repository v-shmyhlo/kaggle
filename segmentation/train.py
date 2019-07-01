import argparse
import gc
import math
import os
import shutil

import numpy as np
import pandas as pd
import torch
import torch.distributions
import torch.utils
import torch.utils.data
import torchvision
import torchvision.datasets
import torchvision.transforms as T
from PIL import Image
from tensorboardX import SummaryWriter
from tqdm import tqdm

import lr_scheduler_wrapper
import utils
from config import Config
# from segmentation.jpu import UNet
from segmentation.transforms import Resize, RandomCrop, CenterCrop, ToTensor, Normalize
from segmentation.unet import UNet

NUM_CLASSES = 150 + 1  # TODO: color, check really has bg
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--config-path', type=str, required=True)
parser.add_argument('--experiment-path', type=str, default='./tf_log/segmentation')
parser.add_argument('--dataset-path', type=str, required=True)
parser.add_argument('--restore-path', type=str)
parser.add_argument('--workers', type=int, default=os.cpu_count())
args = parser.parse_args()
config = Config.from_yaml(args.config_path)
shutil.copy(args.config_path, utils.mkdir(args.experiment_path))

train_transform = T.Compose([
    Resize(config.image_size),
    RandomCrop(config.image_size),
    ToTensor(),
    Normalize(mean=MEAN, std=STD),
])
eval_transform = T.Compose([
    Resize(config.image_size),
    CenterCrop(config.image_size),
    ToTensor(),
    Normalize(mean=MEAN, std=STD),
])


class ADE20K(torch.utils.data.Dataset):
    def __init__(self, path, train, transform):
        subset = 'training' if train else 'validation'
        self.transform = transform

        images = [
            os.path.join(path, 'images', subset, p)
            for p in os.listdir(os.path.join(path, 'images', subset))]
        masks = [
            os.path.join(path, 'annotations', subset, p)
            for p in os.listdir(os.path.join(path, 'annotations', subset))]

        self.data = pd.DataFrame({
            'image_path': sorted(images),
            'mask_path': sorted(masks)
        })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        row = self.data.iloc[item]

        image = Image.open(row['image_path'])
        if image.mode == 'L':
            image = image.convert('RGB')
        mask = Image.open(row['mask_path'])

        if self.transform is not None:
            image, mask = self.transform((image, mask))

        return image, mask


# from shapes import Shapes
#
#
# class ADE20K(Shapes):
#     def __init__(self, path, train, transform):
#         num_samples = 32000 if train else 800
#         super().__init__(num_samples, (config.image_size, config.image_size))
#         self.transform = transform


# class ADE20K(torchvision.datasets.VOCSegmentation):
#     def __init__(self, path, train, transform):
#         subset = 'train' if train else 'val'
#         super().__init__('../data/voc-seg', image_set=subset, download=True)
#         self.tmp = transform
#
#     def __getitem__(self, item):
#         image, mask = super().__getitem__(item)
#
#         if self.tmp is not None:
#             image, mask = self.tmp((image, mask))
#
#         mask[mask == 255] = 0
#
#         return image, mask


# tmp = ADE20K(args.dataset_path, train=True, transform=train_transform)
# # m = max(b[b != 255].max() for a, b in tmp)
# print(m)
# fail


def one_hot(input, n):
    one_hot = torch.zeros(input.size(0), n, input.size(2), input.size(3)).to(input.device)
    input = one_hot.scatter_(1, input, 1)

    return input


def worker_init_fn(_):
    utils.seed_python(torch.initial_seed() % 2**32)


from loss import iou_loss


def compute_loss(input, target):
    input = input.softmax(1)
    target = one_hot(target, NUM_CLASSES)

    loss = iou_loss(input=input, target=target, axis=(2, 3))
    loss = loss.mean(1)

    return loss


# def compute_loss(input, target):
#     target = one_hot(target, NUM_CLASSES)
#     input = input.log_softmax(1)
#     loss = -(target * input).sum(1)
#     loss = loss.mean((1, 2))
#
#     return loss


def draw_masks(input):
    colors = np.random.RandomState(42).uniform(0.25, 1., size=(NUM_CLASSES, 3))
    colors = torch.tensor(colors, dtype=torch.float).to(input.device)
    colors[0, :] = 0.

    input = colors[input]
    input = input.squeeze(1).permute(0, 3, 1, 2)

    return input


def compute_metric(input, target):
    def dice(input, target):
        axis = (2, 3)

        intersection = (input * target).sum(axis)
        union = input.sum(axis) + target.sum(axis)
        v = (2. * intersection) / union
        v[v != v] = 0.
        v = v.mean(1)

        return v

    def iou(input, target):
        axis = (2, 3)

        intersection = (input * target).sum(axis)
        union = input.sum(axis) + target.sum(axis) - intersection
        v = intersection / union
        v[v != v] = 0.
        v = v.mean(1)

        return v

    metric = {
        'dice': dice(input=one_hot(input.argmax(1, keepdim=True), NUM_CLASSES), target=one_hot(target, NUM_CLASSES)),
        'iou': iou(input=one_hot(input.argmax(1, keepdim=True), NUM_CLASSES), target=one_hot(target, NUM_CLASSES)),
    }

    return metric


def build_optimizer(optimizer, parameters):
    if optimizer.type == 'sgd':
        return torch.optim.SGD(
            parameters,
            optimizer.lr,
            momentum=optimizer.sgd.momentum,
            weight_decay=optimizer.weight_decay,
            nesterov=True)
    elif optimizer.type == 'rmsprop':
        return torch.optim.RMSprop(
            parameters,
            optimizer.lr,
            momentum=optimizer.rmsprop.momentum,
            weight_decay=optimizer.weight_decay)
    elif optimizer.type == 'adam':
        return torch.optim.Adam(
            parameters,
            optimizer.lr,
            weight_decay=optimizer.weight_decay)
    else:
        raise AssertionError('invalid OPT {}'.format(optimizer.type))


def train_epoch(model, optimizer, scheduler, data_loader, epoch):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'train'))

    metrics = {
        'loss': utils.Mean(),
    }

    model.train()
    for images, labels in tqdm(data_loader, desc='epoch {} train'.format(epoch)):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)

        loss = compute_loss(input=logits, target=labels)
        metrics['loss'].update(loss.data.cpu().numpy())

        lr, _ = scheduler.get_lr()
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        metrics = {k: metrics[k].compute_and_reset() for k in metrics}
        masks_true = draw_masks(labels)
        masks_pred = draw_masks(logits.argmax(1, keepdim=True))

        print('[EPOCH {}][TRAIN] {}'.format(
            epoch, ', '.join('{}: {:.4f}'.format(k, metrics[k]) for k in metrics)))
        for k in metrics:
            writer.add_scalar(k, metrics[k], global_step=epoch)
        writer.add_scalar('learning_rate', lr, global_step=epoch)
        writer.add_image('images', torchvision.utils.make_grid(
            images, nrow=math.ceil(math.sqrt(images.size(0))), normalize=True), global_step=epoch)
        writer.add_image('masks_true', torchvision.utils.make_grid(
            masks_true, nrow=math.ceil(math.sqrt(masks_true.size(0))), normalize=False), global_step=epoch)
        writer.add_image('masks_pred', torchvision.utils.make_grid(
            masks_pred, nrow=math.ceil(math.sqrt(masks_pred.size(0))), normalize=False), global_step=epoch)


def eval_epoch(model, data_loader, epoch):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'eval'))

    metrics = {
        'loss': utils.Mean(),
        'dice': utils.Mean(),
        'iou': utils.Mean(),
    }

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='epoch {} evaluation'.format(epoch)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)

            loss = compute_loss(input=logits, target=labels)
            metrics['loss'].update(loss.data.cpu().numpy())

            metric = compute_metric(input=logits, target=labels)
            for k in metric:
                metrics[k].update(metric[k].data.cpu().numpy())

        metrics = {k: metrics[k].compute_and_reset() for k in metrics}
        masks_true = draw_masks(labels)
        masks_pred = draw_masks(logits.argmax(1, keepdim=True))

        print('[EPOCH {}][EVAL] {}'.format(
            epoch, ', '.join('{}: {:.4f}'.format(k, metrics[k]) for k in metrics)))
        for k in metrics:
            writer.add_scalar(k, metrics[k], global_step=epoch)
        writer.add_image('images', torchvision.utils.make_grid(
            images, nrow=math.ceil(math.sqrt(images.size(0))), normalize=True), global_step=epoch)
        writer.add_image('masks_true', torchvision.utils.make_grid(
            masks_true, nrow=math.ceil(math.sqrt(masks_true.size(0))), normalize=False), global_step=epoch)
        writer.add_image('masks_pred', torchvision.utils.make_grid(
            masks_pred, nrow=math.ceil(math.sqrt(masks_pred.size(0))), normalize=False), global_step=epoch)

        return metrics


def train():
    train_dataset = ADE20K(args.dataset_path, train=True, transform=train_transform)
    train_dataset = torch.utils.data.Subset(
        train_dataset, np.random.permutation(len(train_dataset))[:len(train_dataset) // 1])
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=args.workers,
        worker_init_fn=worker_init_fn)

    eval_dataset = ADE20K(args.dataset_path, train=False, transform=eval_transform)
    eval_dataset = torch.utils.data.Subset(
        eval_dataset, np.random.permutation(len(eval_dataset))[:len(eval_dataset) // 8])
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        num_workers=args.workers,
        worker_init_fn=worker_init_fn)

    model = UNet(NUM_CLASSES)
    model = model.to(DEVICE)
    if args.restore_path is not None:
        model.load_state_dict(torch.load(args.restore_path))

    optimizer = build_optimizer(config.opt, model.parameters())

    if config.sched.type == 'step':
        scheduler = lr_scheduler_wrapper.EpochWrapper(
            torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=config.sched.step.step_size, gamma=config.sched.step.decay))
    elif config.sched.type == 'plateau':
        scheduler = lr_scheduler_wrapper.ScoreWrapper(
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=config.sched.plateau.decay, patience=config.sched.plateau.patience,
                verbose=True))
    elif config.sched.type == 'cawr':
        scheduler = lr_scheduler_wrapper.StepWrapper(
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=len(train_data_loader), T_mult=2))
    else:
        raise AssertionError('invalid sched {}'.format(config.sched.type))

    for epoch in range(config.epochs):
        train_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data_loader=train_data_loader,
            epoch=epoch)
        gc.collect()
        metric = eval_epoch(
            model=model,
            data_loader=eval_data_loader,
            epoch=epoch)
        gc.collect()

        scheduler.step_epoch()
        scheduler.step_score(metric['iou'])

        torch.save(model.state_dict(), os.path.join(args.experiment_path, 'model.pth'))


def main():
    utils.seed_python(config.seed)
    utils.seed_torch(config.seed)
    train()


if __name__ == '__main__':
    main()
