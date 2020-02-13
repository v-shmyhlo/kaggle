import glob
import importlib.util
import math
import os

import click
import numpy as np
import sklearn
import torch
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter
from tqdm import tqdm

from all_the_tools.metrics import Last, Mean, Metric
from all_the_tools.torch.losses import softmax_cross_entropy
from all_the_tools.torch.optim import LookAhead
from all_the_tools.utils import seed_python
from beng.dataset import LabeledDataset, load_labeled_data, split_target, CLASS_META, decode_target
from beng.model import Model

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class HMAR(Metric):
    def __init__(self):
        self.reset()

    def compute(self):
        input = np.concatenate(self.input, 0)
        target = np.concatenate(self.target, 0)

        scores = [
            sklearn.metrics.recall_score(target[..., i], input[..., i], average='macro')
            for i in range(input.shape[-1])
        ]

        hmar = np.average(scores, weights=CLASS_META['weight'].values)

        return hmar

    def update(self, input, target):
        self.input.append(input)
        self.target.append(target)

    def reset(self):
        self.input = []
        self.target = []


def compute_nrow(images):
    b, _, h, w = images.size()
    nrow = math.ceil(math.sqrt(h * b / w))

    return nrow


def worker_init_fn(_):
    seed_python(torch.initial_seed() % 2**32)


def compute_loss(input, target):
    input = split_target(input, -1)
    target = split_target(target, -1)

    loss = [softmax_cross_entropy(input=i, target=t) for i, t in zip(input, target)]
    loss = sum(loss)

    return loss


def train_epoch(model, data_loader, optimizer, scheduler, epoch, config):
    writer = SummaryWriter(os.path.join(config.experiment_path, 'F{}'.format(config.fold), 'train'))
    metrics = {
        'loss': Mean(),
        'lr': Last(),
    }

    model.train()
    for images, targets in tqdm(data_loader, desc='[F{}][epoch {}] train'.format(config.fold, epoch)):
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        logits = model(images)

        loss = compute_loss(input=logits, target=targets)

        metrics['loss'].update(loss.data.cpu().numpy())
        metrics['lr'].update(np.squeeze(scheduler.get_lr()))

        loss.mean().backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    for k in metrics:
        writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=epoch)
    writer.add_image('images', torchvision.utils.make_grid(
        images, nrow=compute_nrow(images), normalize=True), global_step=epoch)

    writer.flush()
    writer.close()


def eval_epoch(model, data_loader, epoch, config):
    writer = SummaryWriter(os.path.join(config.experiment_path, 'F{}'.format(config.fold), 'eval'))
    metrics = {
        'loss': Mean(),
        'hmar': HMAR(),
    }

    with torch.no_grad():
        model.eval()
        for images, targets in tqdm(data_loader, desc='[F{}][epoch {}] eval'.format(config.fold, epoch)):
            images, targets = images.to(DEVICE), targets.to(DEVICE)

            logits = model(images)

            loss = compute_loss(input=logits, target=targets)

            metrics['loss'].update(loss.data.cpu().numpy())
            metrics['hmar'].update(
                input=decode_target(logits).data.cpu().numpy(),
                target=decode_target(targets).data.cpu().numpy())

    for k in metrics:
        writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=epoch)
    writer.add_image('images', torchvision.utils.make_grid(
        images, nrow=compute_nrow(images), normalize=True), global_step=epoch)

    writer.flush()
    writer.close()


def indices_for_fold(data, fold, seed):
    kfold = KFold(5, shuffle=True, random_state=seed)
    split = list(kfold.split(data))[fold]

    return split


def build_optimizer(parameters, config):
    if config.type == 'sgd':
        optimizer = torch.optim.SGD(
            parameters,
            config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True)
    elif config.type == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            parameters,
            config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay)
    elif config.type == 'adam':
        optimizer = torch.optim.Adam(
            parameters,
            config.lr,
            weight_decay=config.weight_decay)
    else:
        raise AssertionError('invalid optimizer {}'.format(config.type))

    if config.lookahead is not None:
        optimizer = LookAhead(
            optimizer,
            lr=config.lookahead.lr,
            num_steps=config.lookahead.steps)

    # if config.ewa is not None:
    #     optimizer = optim.EWA(
    #         optimizer,
    #         config.ewa.momentum,
    #         num_steps=config.ewa.steps)
    # else:
    #     optimizer = optim.DummySwitchable(optimizer)

    return optimizer


def build_scheduler(optimizer, config, epochs, steps_per_epoch):
    if config.type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * steps_per_epoch)
    else:
        raise AssertionError('invalid scheduler {}'.format(config.type))

    return scheduler


def load_config(config_path, **kwargs):
    spec = importlib.util.spec_from_file_location('config', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.config
    for k in kwargs:
        setattr(config, k, kwargs[k])

    return config


def build_transforms():
    to_tensor_and_norm = T.Compose([
        T.ToTensor(),
        T.Normalize(
            np.mean((0.485, 0.456, 0.406), keepdims=True),
            np.mean((0.229, 0.224, 0.225), keepdims=True)),
    ])
    train_transform = T.Compose([
        to_tensor_and_norm,
    ])
    eval_transform = T.Compose([
        to_tensor_and_norm,
    ])

    return train_transform, eval_transform


@click.command()
@click.option('--config-path', type=click.Path(), required=True)
@click.option('--dataset-path', type=click.Path(), required=True)
@click.option('--experiment-path', type=click.Path(), required=True)
@click.option('--workers', type=click.INT, default=os.cpu_count())
def main(**kwargs):
    # TODO: seed everything

    config = load_config(**kwargs, fold=1)  # FIXME:
    del kwargs

    fold = 0  # FIXME:

    train_eval_data = load_labeled_data(
        os.path.join(config.dataset_path, 'train.csv'),
        glob.glob(os.path.join(config.dataset_path, 'train_image_data_*.parquet')),
        cache_path=os.path.join(config.dataset_path, 'train_images'))

    train_indices, eval_indices = indices_for_fold(train_eval_data, fold=fold, seed=config.seed)

    train_transform, eval_transform = build_transforms()

    train_dataset = LabeledDataset(train_eval_data.iloc[train_indices], transform=train_transform)
    eval_dataset = LabeledDataset(train_eval_data.iloc[eval_indices], transform=eval_transform)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=config.workers,
        worker_init_fn=worker_init_fn)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.eval.batch_size,
        num_workers=config.workers,
        worker_init_fn=worker_init_fn)

    model = Model(config.model, num_classes=CLASS_META['num_classes'].sum()).to(DEVICE)
    optimizer = build_optimizer(model.parameters(), config.train.optimizer)
    scheduler = build_scheduler(optimizer, config.train.scheduler, config.epochs, len(train_data_loader))

    for epoch in range(config.epochs):
        train_epoch(model, train_data_loader, optimizer, scheduler, epoch=epoch, config=config)
        eval_epoch(model, eval_data_loader, epoch=epoch, config=config)


if __name__ == '__main__':
    main()
