import argparse
import gc
import math
import os
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributions
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as T
from PIL import Image
from tensorboardX import SummaryWriter
from tqdm import tqdm

import lr_scheduler_wrapper
import utils
from cells.transforms import RandomFlip, Resize, CenterCrop, RandomCrop, ToTensor
from config import Config
from lr_scheduler import OneCycleScheduler
from metric import accuracy
from .model import Model

# TODO: try largest lr before diverging
# TODO: check all plots rendered
# TODO: better minimum for lr
# TODO: flips
# TODO: grad accum
# TODO: gradient reversal and domain adaptation for test data
# TODO: dropout
# TODO: mixup
# TODO: cell type embedding
# TODO: focal
# TODO: https://www.rxrx.ai/
# TODO: batch effects
# TODO: generalization notes in rxrx
# TODO: metric learning
# TODO: context modelling notes in rxrx


FOLDS = list(range(1, 3 + 1))
NUM_CLASSES = 1108
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--config-path', type=str, required=True)
parser.add_argument('--experiment-path', type=str, default='./tf_log/cells')
parser.add_argument('--dataset-path', type=str, required=True)
parser.add_argument('--restore-path', type=str)
parser.add_argument('--workers', type=int, default=os.cpu_count())
parser.add_argument('--fold', type=int, choices=FOLDS)
args = parser.parse_args()
config = Config.from_yaml(args.config_path)
shutil.copy(args.config_path, utils.mkdir(args.experiment_path))


class RandomSite(object):
    def __call__(self, image):
        if random.random() < 0.5:
            return image[:6]
        else:
            return image[6:]


class SplitInSites(object):
    def __call__(self, image):
        return [image[:6], image[6:]]


train_transform = T.Compose([
    RandomSite(),
    Resize(config.resize_size),
    RandomCrop(config.image_size),
    RandomFlip(),
    ToTensor()
])
eval_transform = T.Compose([
    RandomSite(),
    Resize(config.resize_size),
    CenterCrop(config.image_size),
    ToTensor()
])
test_transform = T.Compose([
    Resize(config.resize_size),
    CenterCrop(config.image_size),
    SplitInSites(),
    T.Lambda(lambda xs: torch.stack([ToTensor()(x) for x in xs], 0))
])


class TrainEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        row = self.data.iloc[item]

        image = []
        for s in [1, 2]:
            image.extend([
                Image.open(os.path.join(
                    row['root'],
                    row['experiment'],
                    'Plate{}'.format(row['plate']),
                    '{}_s{}_w{}.png'.format(row['well'], s, c)))
                for c in range(1, 7)])

        label = row['sirna']
        id = row['id_code']

        if self.transform is not None:
            image = self.transform(image)

        return image, label, id


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        row = self.data.iloc[item]

        image = []
        for s in [1, 2]:
            image.extend([
                Image.open(os.path.join(
                    row['root'],
                    row['experiment'],
                    'Plate{}'.format(row['plate']),
                    '{}_s{}_w{}.png'.format(row['well'], s, c)))
                for c in range(1, 7)])

        id = row['id_code']

        if self.transform is not None:
            image = self.transform(image)

        return image, id


def worker_init_fn(_):
    utils.seed_python(torch.initial_seed() % 2**32)


def compute_loss(input, target):
    loss = F.cross_entropy(input=input, target=target, reduction='none')

    return loss


def compute_metric(input, target):
    metric = {
        'accuracy@1': accuracy(input=input, target=target, topk=1),
        'accuracy@5': accuracy(input=input, target=target, topk=5),
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
            # alpha=0.9999,
            momentum=optimizer.rmsprop.momentum,
            weight_decay=optimizer.weight_decay)
    else:
        raise AssertionError('invalid OPT {}'.format(optimizer.type))


# def indices_for_fold(fold, dataset):
#     kfold = StratifiedKFold(len(FOLDS), shuffle=True, random_state=config.seed)
#     splits = list(kfold.split(np.zeros(len(dataset)), dataset['sirna']))
#     train_indices, eval_indices = splits[fold - 1]
#     assert len(train_indices) + len(eval_indices) == len(dataset)
#
#     # train_indices = train_indices[:len(train_indices) // 1]
#     # eval_indices = eval_indices[:len(eval_indices) // 1]
#
#     return train_indices, eval_indices


# TODO: check
def indices_for_fold(fold, dataset):
    indices = np.arange(len(dataset))
    exp = dataset['experiment']
    eval_exps = \
        ['HEPG2-{:02d}'.format(i + 1) for i in range((fold - 1) * 2, fold * 2)] + \
        ['HUVEC-{:02d}'.format(i + 1) for i in range((fold - 1) * 3, fold * 3)] + \
        ['RPE-{:02d}'.format(i + 1) for i in range((fold - 1) * 2, fold * 2)] + \
        ['U2OS-{:02d}'.format(i + 1) for i in range((fold - 1) * 1, fold * 1)]
    train_indices = indices[~exp.isin(eval_exps)]
    eval_indices = indices[exp.isin(eval_exps)]
    assert np.intersect1d(train_indices, eval_indices).size == 0

    return train_indices, eval_indices


def images_to_rgb(input):
    colors = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ], dtype=np.float32)
    colors = colors / colors.sum(1, keepdims=True)
    print(colors.sum(1))
    colors = colors.reshape((1, 6, 3, 1, 1))
    colors = torch.tensor(colors).to(input.device)
    input = input.unsqueeze(2)
    input = input * colors
    input = input.mean(1)

    return input


def find_lr(train_eval_data):
    train_dataset = TrainEvalDataset(train_eval_data, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=args.workers,
        worker_init_fn=worker_init_fn)

    min_lr = 1e-7
    max_lr = 10.
    gamma = (max_lr / min_lr)**(1 / len(train_data_loader))

    lrs = []
    losses = []
    lim = None

    model = Model(config.model, NUM_CLASSES)
    model = model.to(DEVICE)

    optimizer = build_optimizer(config.opt, model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    model.train()
    for images, labels, ids in tqdm(train_data_loader, desc='lr search'):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)

        loss = compute_loss(input=logits, target=labels)

        lrs.append(np.squeeze(scheduler.get_lr()))
        losses.append(loss.data.cpu().numpy().mean())

        if lim is None:
            lim = losses[0] * 1.1

        if lim < losses[-1]:
            break

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        scheduler.step()

    writer = SummaryWriter(os.path.join(args.experiment_path, 'lr_search'))

    with torch.no_grad():
        losses = np.clip(losses, 0, lim)
        minima_loss = losses[np.argmin(utils.smooth(losses))]
        minima_lr = lrs[np.argmin(utils.smooth(losses))]

        step = 0
        for loss, loss_sm in zip(losses, utils.smooth(losses)):
            writer.add_scalar('search_loss', loss, global_step=step)
            writer.add_scalar('search_loss_sm', loss_sm, global_step=step)
            step += config.batch_size

        fig = plt.figure()
        plt.plot(lrs, losses)
        plt.plot(lrs, utils.smooth(losses))
        plt.axvline(minima_lr)
        plt.xscale('log')
        plt.title('loss: {:.8f}, lr: {:.8f}'.format(minima_loss, minima_lr))
        writer.add_figure('search', fig, global_step=0)

        return minima_lr


def train_epoch(model, optimizer, scheduler, data_loader, fold, epoch):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'fold{}'.format(fold), 'train'))

    metrics = {
        'loss': utils.Mean(),
    }

    model.train()
    for images, labels, ids in tqdm(data_loader, desc='epoch {} train'.format(epoch)):
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
        images = images_to_rgb(images)[:16]
        print('[EPOCH {}][TRAIN] {}'.format(
            epoch, ', '.join('{}: {:.4f}'.format(k, metrics[k]) for k in metrics)))
        for k in metrics:
            writer.add_scalar(k, metrics[k], global_step=epoch)
        writer.add_scalar('learning_rate', lr, global_step=epoch)
        writer.add_image('images', torchvision.utils.make_grid(
            images, nrow=math.ceil(math.sqrt(images.size(0))), normalize=True), global_step=epoch)


def eval_epoch(model, data_loader, fold, epoch):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'fold{}'.format(fold), 'eval'))

    metrics = {
        'loss': utils.Mean(),
        'accuracy@1': utils.Mean(),
        'accuracy@5': utils.Mean(),
    }

    model.eval()
    with torch.no_grad():
        for images, labels, ids in tqdm(data_loader, desc='epoch {} evaluation'.format(epoch)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)

            loss = compute_loss(input=logits, target=labels)
            metrics['loss'].update(loss.data.cpu().numpy())

            metric = compute_metric(input=logits, target=labels)
            for k in metric:
                metrics[k].update(metric[k].data.cpu().numpy())

        metrics = {k: metrics[k].compute_and_reset() for k in metrics}
        images = images_to_rgb(images)[:16]
        print('[EPOCH {}][EVAL] {}'.format(
            epoch, ', '.join('{}: {:.4f}'.format(k, metrics[k]) for k in metrics)))
        for k in metrics:
            writer.add_scalar(k, metrics[k], global_step=epoch)
        writer.add_image('images', torchvision.utils.make_grid(
            images, nrow=math.ceil(math.sqrt(images.size(0))), normalize=True), global_step=epoch)

        return metrics


def train_fold(fold, train_eval_data):
    train_indices, eval_indices = indices_for_fold(fold, train_eval_data)

    train_dataset = TrainEvalDataset(train_eval_data.iloc[train_indices], transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=args.workers,
        worker_init_fn=worker_init_fn)
    eval_dataset = TrainEvalDataset(train_eval_data.iloc[eval_indices], transform=eval_transform)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        drop_last=True,
        num_workers=args.workers,
        worker_init_fn=worker_init_fn)

    model = Model(config.model, NUM_CLASSES)
    model = model.to(DEVICE)
    if args.restore_path is not None:
        model.load_state_dict(torch.load(os.path.join(args.restore_path, 'model_{}.pth'.format(fold))))

    optimizer = build_optimizer(config.opt, model.parameters())

    if config.sched.type == 'onecycle':
        scheduler = lr_scheduler_wrapper.StepWrapper(
            OneCycleScheduler(
                optimizer,
                lr=(config.opt.lr / 20, config.opt.lr),
                beta=config.sched.onecycle.beta,
                max_steps=len(train_data_loader) * config.epochs,
                annealing=config.sched.onecycle.anneal))
    elif config.sched.type == 'cyclic':
        step_size_up = len(train_data_loader) * config.sched.cyclic.step_size_up
        step_size_down = len(train_data_loader) * config.sched.cyclic.step_size_down

        scheduler = lr_scheduler_wrapper.StepWrapper(
            torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                0.,
                config.opt.lr,
                step_size_up=step_size_up,
                step_size_down=step_size_down,
                mode='exp_range',
                gamma=config.sched.cyclic.decay**(1 / (step_size_up + step_size_down)),
                cycle_momentum=True,
                base_momentum=0.75,
                max_momentum=0.95))
    elif config.sched.type == 'cawr':
        scheduler = lr_scheduler_wrapper.StepWrapper(
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=len(train_data_loader), T_mult=2))
    elif config.sched.type == 'plateau':
        scheduler = lr_scheduler_wrapper.ScoreWrapper(
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=0, verbose=True))
    else:
        raise AssertionError('invalid sched {}'.format(config.sched.type))

    best_score = 0
    for epoch in range(config.epochs):
        train_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data_loader=train_data_loader,
            fold=fold,
            epoch=epoch)
        gc.collect()
        metric = eval_epoch(
            model=model,
            data_loader=eval_data_loader,
            fold=fold,
            epoch=epoch)
        gc.collect()

        score = metric['accuracy@1']

        scheduler.step_epoch()
        scheduler.step_score(score)

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(args.experiment_path, 'model_{}.pth'.format(fold)))


def build_submission(folds, test_data):
    with torch.no_grad():
        predictions = 0.

        for fold in folds:
            fold_predictions, fold_ids = predict_on_test_using_fold(fold, test_data)
            fold_predictions = fold_predictions.softmax(2).mean(1)

            predictions = predictions + fold_predictions
            ids = fold_ids

        predictions = predictions / len(folds)
        predictions = predictions.argmax(1).data.cpu().numpy()
        assert len(ids) == len(predictions)

        submission = pd.DataFrame({'id_code': ids, 'sirna': predictions})
        submission.to_csv(os.path.join(args.experiment_path, 'submission.csv'), index=False)
        submission.to_csv('./submission.csv', index=False)


def predict_on_test_using_fold(fold, test_data):
    test_dataset = TestDataset(test_data, transform=test_transform)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
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
            logits = logits.view(b, n, NUM_CLASSES)

            fold_predictions.append(logits)
            fold_ids.extend(ids)

        fold_predictions = torch.cat(fold_predictions, 0)

    return fold_predictions, fold_ids


def main():
    utils.seed_python(config.seed)
    utils.seed_torch(config.seed)

    train_eval_data = pd.read_csv(os.path.join(args.dataset_path, 'train.csv'))
    train_eval_data['root'] = os.path.join(args.dataset_path, 'train')

    test_data = pd.read_csv(os.path.join(args.dataset_path, 'test.csv'))
    test_data['root'] = os.path.join(args.dataset_path, 'test')

    if config.opt.lr is None:
        lr = find_lr(train_eval_data)
        print('find_lr: {}'.format(lr))
        gc.collect()
        fail  # FIXME:

    if args.fold is None:
        folds = FOLDS
    else:
        folds = [args.fold]

    for fold in folds:
        train_fold(fold, train_eval_data)

    build_submission(folds, test_data)


if __name__ == '__main__':
    main()
