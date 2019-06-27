import argparse
import gc
import math
import os
import shutil

import torch
import torch.distributions
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torchvision
import torchvision.datasets
import torchvision.transforms as T
from tensorboardX import SummaryWriter
from tqdm import tqdm

import lr_scheduler_wrapper
import utils
from classification.mobilenet_v3 import MobileNetV3
from config import Config
from metric import accuracy

# TODO: dropout

NUM_CLASSES = 1000
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--config-path', type=str, required=True)
parser.add_argument('--experiment-path', type=str, default='./tf_log/detection')
parser.add_argument('--dataset-path', type=str, required=True)
parser.add_argument('--restore-path', type=str)
parser.add_argument('--workers', type=int, default=os.cpu_count())
args = parser.parse_args()
config = Config.from_yaml(args.config_path)
shutil.copy(args.config_path, utils.mkdir(args.experiment_path))

train_transform = T.Compose([
    T.RandomSizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),

    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])
eval_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),

    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


def worker_init_fn(_):
    utils.seed_python(torch.initial_seed() % 2**32)


def compute_loss(input, target):
    loss = F.cross_entropy(input=input, target=target, reduction='none')

    return loss


def compute_metric(input, target):
    metric = {
        'accuracy@1': accuracy(input=input, target=target, topk=1),
        'accuracy@5': accuracy(input=input, target=target, topk=5)
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
            alpha=0.99,
            momentum=optimizer.rmsprop.momentum,
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

        print('[EPOCH {}][TRAIN] {}'.format(
            epoch, ', '.join('{}: {:.4f}'.format(k, metrics[k]) for k in metrics)))
        for k in metrics:
            writer.add_scalar(k, metrics[k], global_step=epoch)
        writer.add_scalar('learning_rate', lr, global_step=epoch)
        writer.add_image('images', torchvision.utils.make_grid(
            images[:32], nrow=math.ceil(math.sqrt(images[:32].size(0))), normalize=True), global_step=epoch)


def eval_epoch(model, data_loader, epoch):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'eval'))

    metrics = {
        'loss': utils.Mean(),
        'accuracy@1': utils.Mean(),
        'accuracy@5': utils.Mean()
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

        print('[EPOCH {}][EVAL] {}'.format(
            epoch, ', '.join('{}: {:.4f}'.format(k, metrics[k]) for k in metrics)))
        for k in metrics:
            writer.add_scalar(k, metrics[k], global_step=epoch)
        writer.add_image('images', torchvision.utils.make_grid(
            images[:32], nrow=math.ceil(math.sqrt(images[:32].size(0))), normalize=True), global_step=epoch)

        return metrics


def train():
    train_dataset = torchvision.datasets.ImageNet(
        args.dataset_path, split='train', transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=args.workers,
        worker_init_fn=worker_init_fn)

    eval_dataset = torchvision.datasets.ImageNet(
        args.dataset_path, split='val', transform=eval_transform)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.batch_size * 2,
        num_workers=args.workers,
        worker_init_fn=worker_init_fn)

    model = MobileNetV3(3, NUM_CLASSES)
    model = model.to(DEVICE)
    if args.restore_path is not None:
        model.load_state_dict(torch.load(args.restore_path))

    optimizer = build_optimizer(config.opt, model.parameters())

    if config.sched.type == 'step':
        scheduler = lr_scheduler_wrapper.EpochWrapper(
            torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.sched.step.step_size,
                gamma=config.sched.step.decay))
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
        scheduler.step_score(metric['accuracy@1'])

        torch.save(model.state_dict(), os.path.join(args.experiment_path, 'model.pth'))


def main():
    utils.seed_python(config.seed)
    utils.seed_torch(config.seed)
    train()


if __name__ == '__main__':
    main()
