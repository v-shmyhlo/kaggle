import argparse
import gc
import math
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributions
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as T
from tensorboardX import SummaryWriter
from tqdm import tqdm

import lr_scheduler_wrapper
import optim
import utils
from config import Config
from loss import dice_loss
from lr_scheduler import OneCycleScheduler
from stal.dataset import NUM_CLASSES, TrainEvalDataset, TestDataset
from stal.model import Model
from stal.transforms import RandomCrop, CenterCrop, ApplyTo, Extract
from stal.utils import mask_to_image

FOLDS = list(range(1, 5 + 1))
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--config-path', type=str, required=True)
parser.add_argument('--experiment-path', type=str, default='./tf_log/stal')
parser.add_argument('--dataset-path', type=str, required=True)
parser.add_argument('--restore-path', type=str)
parser.add_argument('--workers', type=int, default=os.cpu_count())
parser.add_argument('--fold', type=int, choices=FOLDS)
parser.add_argument('--infer', action='store_true')
parser.add_argument('--lr-search', action='store_true')
args = parser.parse_args()
config = Config.from_yaml(args.config_path)
shutil.copy(args.config_path, utils.mkdir(args.experiment_path))


# assert config.resize_size == config.crop_size


class Resetable(object):
    def __init__(self, build_transform):
        self.build_transform = build_transform

    def __call__(self, input):
        return self.transform(input)

    def reset(self, *args, **kwargs):
        self.transform = self.build_transform(*args, **kwargs)


class RandomResize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, input):
        size = round(np.random.uniform(self.min_size, self.max_size))
        input = Resize(size)(input)

        return input


# random_resize = Resetable(RandomResize)
# random_crop = Resetable(RandomCrop)
# center_crop = Resetable(CenterCrop)
# to_tensor = ToTensor()

# if config.normalize:
#     normalize = NormalizeByExperimentStats(
#         torch.load('./experiment_stats.pth'))  # TODO: needs realtime computation on private
# else:
#     normalize = T.Compose([])


train_transform = T.Compose([
    RandomCrop((256, 1024)),
    ApplyTo(
        ['image', 'mask'],
        T.ToTensor()),
    ApplyTo(
        ['mask'],
        T.Lambda(lambda x: x.long())),
    Extract(['image', 'mask', 'id']),
])
eval_transform = T.Compose([
    CenterCrop((256, 1024)),
    ApplyTo(
        ['image', 'mask'],
        T.ToTensor()),
    ApplyTo(
        ['mask'],
        T.Lambda(lambda x: x.long())),
    Extract(['image', 'mask', 'id']),
])
test_transform = T.Compose([
    ApplyTo(
        ['image'],
        T.ToTensor()),
    Extract(['image', 'id']),
])


def update_transforms(p):
    assert 0. <= p <= 1.

    # crop_size = round(224 + (config.crop_size - 224) * p)
    # delta = config.resize_size - crop_size
    # resize_size = config.resize_size - delta, config.resize_size + delta
    # assert sum(resize_size) / 2 == config.resize_size
    # print('update transforms p: {:.2f}, resize_size: {}, crop_size: {}'.format(p, resize_size, crop_size))
    #
    # random_resize.reset(*resize_size)
    # random_crop.reset(crop_size)
    # center_crop.reset(crop_size)


# TODO: use pool
def find_temp_global(input, target, exps):
    temps = np.logspace(np.log(1e-4), np.log(1.0), 50, base=np.e)
    metrics = []
    for temp in tqdm(temps, desc='temp search'):
        fold_preds = assign_classes(probs=(input * temp).softmax(1).data.cpu().numpy(), exps=exps)
        fold_preds = torch.tensor(fold_preds).to(input.device)
        metric = compute_metric(input=fold_preds, target=target)
        metrics.append(metric['dice'].mean().data.cpu().numpy())

    temp = temps[np.argmax(metrics)]
    metric = metrics[np.argmax(metrics)]
    fig = plt.figure()
    plt.plot(temps, metrics)
    plt.xscale('log')
    plt.axvline(temp)
    plt.title('metric: {:.4f}, temp: {:.4f}'.format(metric.item(), temp))
    plt.savefig('./fig.png')

    return temp, metric.item(), fig


def worker_init_fn(_):
    utils.seed_python(torch.initial_seed() % 2**32)


def mixup(images_1, labels_1, ids, alpha):
    dist = torch.distributions.beta.Beta(alpha, alpha)
    indices = np.random.permutation(len(ids))
    images_2, labels_2 = images_1[indices], labels_1[indices]

    lam = dist.sample().to(DEVICE)
    lam = torch.max(lam, 1 - lam)

    images = lam * images_1.to(DEVICE) + (1 - lam) * images_2.to(DEVICE)
    labels = lam * labels_1.to(DEVICE) + (1 - lam) * labels_2.to(DEVICE)

    return images, labels, ids


def compute_nrow(images):
    b, _, h, w = images.size()
    nrow = math.ceil(math.sqrt(h * b / w))

    return nrow


def focal_loss(input, target, gamma=2.):
    axis = 1

    prob = input.softmax(axis)
    weight = (1 - prob)**gamma
    loss = -(weight * target * input.log_softmax(axis)).sum(axis)

    return loss


def compute_loss(input, target):
    target = utils.one_hot(target.squeeze(1), num_classes=NUM_CLASSES).permute((0, 3, 1, 2))
    loss = focal_loss(input=input, target=target)
    loss = loss.mean((1, 2))

    return loss


# def compute_loss(input, target):
#     # TODO: check loss
#
#     input = input.softmax(1)
#     target = utils.one_hot(target.squeeze(1), num_classes=NUM_CLASSES).permute((0, 3, 1, 2))
#
#     # input, target = input[:, 1:], target[:, 1:]
#
#     loss = dice_loss(input=input, target=target, axis=(2, 3))
#
#     return loss


def compute_metric(input, target):
    input = utils.one_hot(input.argmax(1), num_classes=NUM_CLASSES).permute((0, 3, 1, 2))
    target = utils.one_hot(target.squeeze(1), num_classes=NUM_CLASSES).permute((0, 3, 1, 2))

    input, target = input[:, 1:], target[:, 1:]

    dice = dice_loss(input=input, target=target, axis=(2, 3), eps=0.)
    dice = (-dice).exp()
    dice[dice != dice] = 0.
    # dice[(input.sum((2, 3)) == 0.) & (target.sum((2, 3)) == 0.)] = 1.

    metric = {
        'dice': dice
    }

    return metric


def build_optimizer(optimizer_config, parameters):
    if optimizer_config.type == 'sgd':
        optimizer = torch.optim.SGD(
            parameters,
            optimizer_config.lr,
            momentum=optimizer_config.sgd.momentum,
            weight_decay=optimizer_config.weight_decay,
            nesterov=True)
    elif optimizer_config.type == 'adam':
        optimizer = torch.optim.Adam(
            parameters,
            optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay)
    elif optimizer_config.type == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            parameters,
            optimizer_config.lr,
            momentum=optimizer_config.rmsprop.momentum,
            weight_decay=optimizer_config.weight_decay)
    else:
        raise AssertionError('invalid OPT {}'.format(optimizer_config.type))

    if optimizer_config.lookahead is not None:
        optimizer = optim.LA(
            optimizer,
            optimizer_config.lookahead.lr,
            num_steps=optimizer_config.lookahead.steps)

    return optimizer


# def indices_for_fold(fold, dataset):
#     kfold = KFold(len(FOLDS), shuffle=True, random_state=config.seed)
#     splits = list(kfold.split(list(range(len(dataset)))))
#     train_indices, eval_indices = splits[fold - 1]
#     print(len(train_indices), len(eval_indices))
#     assert len(train_indices) + len(eval_indices) == len(dataset)
#
#     return train_indices, eval_indices


def indices_for_fold(fold, dataset):
    ids = dataset['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

    unique_ids = ids.unique()
    unique_ids = unique_ids[np.random.RandomState(42).permutation(len(unique_ids))]
    train_ids, eval_ids = unique_ids[len(unique_ids) // 5:], unique_ids[:len(unique_ids) // 5]

    indices = np.arange(len(dataset))
    train_indices, eval_indices = indices[ids.isin(train_ids)], indices[ids.isin(eval_ids)]

    print(len(train_indices), len(eval_indices))

    return train_indices, eval_indices


def lr_search(train_eval_data):
    train_eval_dataset = TrainEvalDataset(train_eval_data, transform=train_transform)
    train_eval_data_loader = torch.utils.data.DataLoader(
        train_eval_dataset,
        batch_size=config.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=args.workers,
        worker_init_fn=worker_init_fn)

    min_lr = 1e-7
    max_lr = 10.
    gamma = (max_lr / min_lr)**(1 / len(train_eval_data_loader))

    lrs = []
    losses = []
    lim = None

    model = Model(config.model, NUM_CLASSES)
    model = model.to(DEVICE)

    optimizer = build_optimizer(config.opt, model.parameters())
    for param_group in optimizer.param_groups:
        param_group['lr'] = min_lr
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    update_transforms(1.)
    model.train()
    optimizer.zero_grad()
    for i, (images, masks, _) in enumerate(tqdm(train_eval_data_loader, desc='lr search'), 1):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        logits = model(images)

        loss = compute_loss(input=logits, target=masks)

        lrs.append(np.squeeze(scheduler.get_lr()))
        losses.append(loss.data.cpu().numpy().mean())

        if lim is None:
            lim = losses[0] * 1.1

        if lim < losses[-1]:
            break

        (loss.mean() / config.opt.acc_steps).backward()

        if i % config.opt.acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

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

    update_transforms(np.linspace(0, 1, config.epochs)[epoch - 1].item())
    model.train()
    optimizer.zero_grad()
    for i, (images, masks, ids) in enumerate(tqdm(data_loader, desc='epoch {} train'.format(epoch)), 1):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        logits = model(images)

        loss = compute_loss(input=logits, target=masks)
        metrics['loss'].update(loss.data.cpu().numpy())

        lr = scheduler.get_lr()
        (loss.mean() / config.opt.acc_steps).backward()

        if i % config.opt.acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

    with torch.no_grad():
        metrics = {k: metrics[k].compute_and_reset() for k in metrics}
        print('[FOLD {}][EPOCH {}][TRAIN] {}'.format(
            fold, epoch, ', '.join('{}: {:.4f}'.format(k, metrics[k]) for k in metrics)))
        for k in metrics:
            writer.add_scalar(k, metrics[k], global_step=epoch)
        writer.add_scalar('learning_rate', lr, global_step=epoch)

        images = images[:32]
        masks = mask_to_image(masks[:32], num_classes=NUM_CLASSES)
        preds = mask_to_image(logits[:32].argmax(1, keepdim=True), num_classes=NUM_CLASSES)

        writer.add_image('images', torchvision.utils.make_grid(
            images, nrow=compute_nrow(images), normalize=True), global_step=epoch)
        writer.add_image('masks', torchvision.utils.make_grid(
            masks, nrow=compute_nrow(masks), normalize=True), global_step=epoch)
        writer.add_image('preds', torchvision.utils.make_grid(
            preds, nrow=compute_nrow(preds), normalize=True), global_step=epoch)


def eval_epoch(model, data_loader, fold, epoch):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'fold{}'.format(fold), 'eval'))

    metrics = {
        'loss': utils.Mean(),
        'dice': utils.Mean(),
    }

    model.eval()
    with torch.no_grad():
        for images, masks, _ in tqdm(data_loader, desc='epoch {} evaluation'.format(epoch)):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            logits = model(images)

            loss = compute_loss(input=logits, target=masks)
            metrics['loss'].update(loss.data.cpu().numpy())

            metric = compute_metric(input=logits, target=masks)
            for k in metric:
                metrics[k].update(metric[k].data.cpu().numpy())

        metrics = {k: metrics[k].compute_and_reset() for k in metrics}
        print('[FOLD {}][EPOCH {}][EVAL] {}'.format(
            fold, epoch, ', '.join('{}: {:.4f}'.format(k, metrics[k]) for k in metrics)))
        for k in metrics:
            writer.add_scalar(k, metrics[k], global_step=epoch)

        images = images[:32]
        masks = mask_to_image(masks[:32], num_classes=NUM_CLASSES)
        preds = mask_to_image(logits[:32].argmax(1, keepdim=True), num_classes=NUM_CLASSES)

        writer.add_image('images', torchvision.utils.make_grid(
            images, nrow=compute_nrow(images), normalize=True), global_step=epoch)
        writer.add_image('masks', torchvision.utils.make_grid(
            masks, nrow=compute_nrow(masks), normalize=True), global_step=epoch)
        writer.add_image('preds', torchvision.utils.make_grid(
            preds, nrow=compute_nrow(preds), normalize=True), global_step=epoch)

        return metrics


def train_fold(fold, train_eval_data):
    train_indices, eval_indices = indices_for_fold(fold, train_eval_data)  # FIXME: dataset size

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
                annealing=config.sched.onecycle.anneal,
                peak_pos=config.sched.onecycle.peak_pos,
                end_pos=config.sched.onecycle.end_pos))
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
                optimizer,
                mode='max',
                factor=config.sched.plateau.decay,
                patience=config.sched.plateau.patience,
                verbose=True))
    else:
        raise AssertionError('invalid sched {}'.format(config.sched.type))

    best_score = 0
    for epoch in range(1, config.epochs + 1):
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

        score = metric['dice']

        scheduler.step_epoch()
        scheduler.step_score(score)

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(args.experiment_path, 'model_{}.pth'.format(fold)))


def build_submission(folds, test_data, temp):
    with torch.no_grad():
        probs = 0.

        for fold in folds:
            fold_logits, fold_exps, fold_ids = predict_on_test_using_fold(fold, test_data)
            fold_probs = (fold_logits * temp).softmax(2).mean(1)

            probs = probs + fold_probs
            exps = fold_exps
            ids = fold_ids

        probs = probs / len(folds)
        probs = probs.data.cpu().numpy()
        assert len(probs) == len(exps) == len(ids)
        classes = assign_classes(probs=probs, exps=exps)

        submission = pd.DataFrame({'id_code': ids, 'sirna': classes})
        submission.to_csv(os.path.join(args.experiment_path, 'submission.csv'), index=False)
        submission.to_csv('./submission.csv', index=False)


def predict_on_test_using_fold(fold, test_data):
    test_dataset = TestDataset(test_data, transform=test_transform)
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
        fold_logits = []
        fold_exps = []
        fold_ids = []

        for images, feats, exps, ids in tqdm(test_data_loader, desc='fold {} inference'.format(fold)):
            images, feats = images.to(DEVICE), feats.to(DEVICE)

            b, n, c, h, w = images.size()
            images = images.view(b * n, c, h, w)
            feats = feats.view(b, 1, 2).repeat(1, n, 1).view(b * n, 2)
            logits = model(images, feats)
            logits = logits.view(b, n, NUM_CLASSES)

            fold_logits.append(logits)
            fold_exps.extend(exps)
            fold_ids.extend(ids)

        fold_logits = torch.cat(fold_logits, 0)

    torch.save((fold_logits, fold_exps, fold_ids), './test_{}.pth'.format(fold))

    return fold_logits, fold_exps, fold_ids


def predict_on_eval_using_fold(fold, train_eval_data):
    _, eval_indices = indices_for_fold(fold, train_eval_data)
    eval_data = train_eval_data.iloc[eval_indices]
    eval_dataset = TrainEvalDataset(eval_data, transform=eval_transform)
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
        fold_labels = []
        fold_logits = []
        fold_exps = []
        fold_ids = []

        for images, feats, exps, labels, ids in tqdm(eval_data_loader, desc='fold {} evaluation'.format(fold)):
            images, feats, labels = images.to(DEVICE), feats.to(DEVICE), labels.to(DEVICE)
            logits = model(images, feats)

            fold_labels.append(labels)
            fold_logits.append(logits)
            fold_exps.extend(exps)
            fold_ids.extend(ids)

        fold_labels = torch.cat(fold_labels, 0)
        fold_logits = torch.cat(fold_logits, 0)

        return fold_labels, fold_logits, fold_exps, fold_ids


def find_temp_for_folds(folds, train_eval_data):
    with torch.no_grad():
        labels = []
        logits = []
        exps = []
        ids = []

        for fold in folds:
            fold_labels, fold_logits, fold_exps, fold_ids = predict_on_eval_using_fold(fold, train_eval_data)

            labels.append(fold_labels)
            logits.append(fold_logits)
            exps.extend(fold_exps)
            ids.extend(fold_ids)

        labels = torch.cat(labels, 0)
        logits = torch.cat(logits, 0)

        temp, metric, _ = find_temp_global(input=logits, target=labels, exps=exps)
        print('metric: {:.4f}, temp: {:.4f}'.format(metric, temp))
        torch.save((labels, logits, exps, ids), './oof.pth')

        return temp


def main():
    utils.seed_python(config.seed)
    utils.seed_torch(config.seed)

    train_eval_data = pd.read_csv(os.path.join(args.dataset_path, 'train.csv'), converters={'EncodedPixels': str})
    train_eval_data['root'] = os.path.join(args.dataset_path, 'train_images')

    test_data = pd.read_csv(os.path.join(args.dataset_path, 'sample_submission.csv'), converters={'EncodedPixels': str})
    test_data['root'] = os.path.join(args.dataset_path, 'test_images')

    if args.lr_search:
        lr = lr_search(train_eval_data)
        print('lr_search: {}'.format(lr))
        gc.collect()
        return

    if args.fold is None:
        folds = FOLDS
    else:
        folds = [args.fold]

    if not args.infer:
        for fold in folds:
            train_fold(fold, train_eval_data)

    update_transforms(1.)  # FIXME:
    temp = find_temp_for_folds(folds, train_eval_data)
    gc.collect()
    build_submission(folds, test_data, temp)


if __name__ == '__main__':
    main()
