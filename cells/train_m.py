import argparse
import gc
import math
import os
import shutil

import lap
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
from tensorboardX import SummaryWriter
from tqdm import tqdm

import lr_scheduler_wrapper
import utils
from cells.dataset import NUM_CLASSES, TrainEvalDataset, TestDataset
from cells.model_m import Model
from cells.transforms import Extract, ApplyTo, RandomFlip, RandomTranspose, Resize, ToTensor, RandomSite, SplitInSites, \
    ChannelReweight, RandomRotation, RandomCrop, CenterCrop
from cells.utils import images_to_rgb
from config import Config
from lr_scheduler import OneCycleScheduler

FOLDS = list(range(1, 3 + 1))
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--config-path', type=str, required=True)
parser.add_argument('--experiment-path', type=str, default='./tf_log/cells')
parser.add_argument('--dataset-path', type=str, required=True)
parser.add_argument('--restore-path', type=str)
parser.add_argument('--workers', type=int, default=os.cpu_count())
parser.add_argument('--fold', type=int, choices=FOLDS)
parser.add_argument('--infer', action='store_true')
parser.add_argument('--lr-search', action='store_true')
args = parser.parse_args()
config = Config.from_yaml(args.config_path)
shutil.copy(args.config_path, utils.mkdir(args.experiment_path))
assert config.resize_size == config.crop_size


class Resetable(object):
    def __init__(self, build_transform):
        self.build_transform = build_transform

    def __call__(self, input):
        return self.transform(input)

    def reset(self, *args, **kwargs):
        self.transform = self.build_transform(*args, **kwargs)


random_crop = Resetable(RandomCrop)
center_crop = Resetable(CenterCrop)

train_transform = T.Compose([
    ApplyTo(
        ['image'],
        T.Compose([
            RandomSite(),
            Resize(config.resize_size),
            random_crop,
            RandomFlip(),
            RandomTranspose(),
            RandomRotation(180),  # FIXME:
            ToTensor(),
            ChannelReweight(config.aug.channel_weight),
        ])),
    # NormalizeByRefStats(),
    Extract(['image', 'feat', 'label', 'id']),
])
eval_transform = T.Compose([
    ApplyTo(
        ['image'],
        T.Compose([
            RandomSite(),  # FIXME:
            Resize(config.resize_size),
            center_crop,
            ToTensor(),
        ])),
    # NormalizeByRefStats(),
    Extract(['image', 'feat', 'label', 'id']),
])
test_transform = T.Compose([
    ApplyTo(
        ['image'],
        T.Compose([
            Resize(config.resize_size),
            center_crop,
            SplitInSites(),
            T.Lambda(lambda xs: torch.stack([ToTensor()(x) for x in xs], 0)),
        ])),
    # NormalizeByRefStats(),
    Extract(['image', 'feat', 'id']),
])


def update_transforms(image_size):
    print('update transforms: {}'.format(image_size))
    random_crop.reset(image_size)
    center_crop.reset(image_size)


# TODO: use pool
def find_temp_global(input, target, data):
    temps = np.logspace(np.log(0.001), np.log(1.), 30, base=np.e)
    metrics = []
    for temp in tqdm(temps, desc='temp search'):
        fold_preds = assign_classes(probs=(input * temp).softmax(1).data.cpu().numpy(), data=data)
        fold_preds = torch.tensor(fold_preds).to(input.device)
        metric = compute_metric(input=fold_preds, target=target)
        metrics.append(metric['accuracy@1'].mean().data.cpu().numpy())

    temp = temps[np.argmax(metrics)]
    metric = metrics[np.argmax(metrics)]
    fig = plt.figure()
    plt.plot(temps, metrics)
    plt.xscale('log')
    plt.axvline(temp)
    plt.title('metric: {:.4f}, temp: {:.4f}'.format(metric.item(), temp))

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


def compute_loss(input, target, weight=1.0):
    input, arc_input = input

    loss = F.cross_entropy(input=input, target=target, reduction='none')

    if arc_input is not None:
        arc_loss = F.cross_entropy(input=arc_input, target=target, reduction='none')
        loss = loss * weight + arc_loss * (1 - weight)

    return loss


def compute_metric(input, target):
    metric = {
        'accuracy@1': (input == target).float(),
    }

    return metric


def assign_classes(probs, data):
    # TODO: refactor numpy/torch usage

    classes = np.zeros(probs.shape[0], dtype=np.int64)
    for exp in data['experiment'].unique():
        subset = data['experiment'] == exp
        preds = probs[subset]
        _, c, _ = lap.lapjv(1 - preds, extend_cost=True)
        classes[subset] = c

    return classes


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
    else:
        raise AssertionError('invalid OPT {}'.format(optimizer.type))


def indices_for_fold(fold, dataset):
    fold_eval_exps = [
        None,
        ['HEPG2-02', 'HEPG2-05', 'HUVEC-12', 'HUVEC-09', 'HUVEC-03', 'HUVEC-07', 'HUVEC-01', 'RPE-01', 'RPE-04',
         'U2OS-01'],
        ['HEPG2-03', 'HEPG2-06', 'HUVEC-13', 'HUVEC-10', 'HUVEC-06', 'HUVEC-11', 'HUVEC-02', 'RPE-02', 'RPE-06',
         'U2OS-02'],
        ['HEPG2-04', 'HEPG2-07', 'HUVEC-16', 'HUVEC-14', 'HUVEC-08', 'HUVEC-15', 'HUVEC-04', 'RPE-03', 'RPE-07',
         'U2OS-03'],
    ]

    indices = np.arange(len(dataset))
    exp = dataset['experiment']
    eval_exps = fold_eval_exps[fold]
    train_indices = indices[~exp.isin(eval_exps)]
    eval_indices = indices[exp.isin(eval_exps)]
    assert np.intersect1d(train_indices, eval_indices).size == 0
    assert round(len(train_indices) / len(eval_indices), 1) == 2.3

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

    update_transforms(config.crop_size)
    model.train()
    optimizer.zero_grad()
    for i, (images, feats, labels, ids) in enumerate(tqdm(train_eval_data_loader, desc='lr search'), 1):
        images, feats, labels = images.to(DEVICE), feats.to(DEVICE), labels.to(DEVICE)
        logits = model(images, feats, labels)

        loss = compute_loss(input=logits, target=labels)
        logits, _ = logits

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

    update_transforms(round(224 + (config.crop_size - 224) * np.linspace(0, 1, config.epochs)[epoch - 1].item()))
    model.train()
    optimizer.zero_grad()
    for i, (images, feats, labels, ids) in enumerate(tqdm(data_loader, desc='epoch {} train'.format(epoch)), 1):
        images, feats, labels = images.to(DEVICE), feats.to(DEVICE), labels.to(DEVICE)
        logits = model(images, feats, labels)

        loss = compute_loss(input=logits, target=labels, weight=np.linspace(1., 0.8, config.epochs)[epoch - 1])
        logits, _ = logits
        metrics['loss'].update(loss.data.cpu().numpy())

        lr = scheduler.get_lr()
        (loss.mean() / config.opt.acc_steps).backward()

        if i % config.opt.acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

    with torch.no_grad():
        metrics = {k: metrics[k].compute_and_reset() for k in metrics}
        images = images_to_rgb(images)[:16]
        print('[FOLD {}][EPOCH {}][TRAIN] {}'.format(
            fold, epoch, ', '.join('{}: {:.4f}'.format(k, metrics[k]) for k in metrics)))
        for k in metrics:
            writer.add_scalar(k, metrics[k], global_step=epoch)
        writer.add_scalar('learning_rate', lr, global_step=epoch)
        writer.add_image('images', torchvision.utils.make_grid(
            images, nrow=math.ceil(math.sqrt(images.size(0))), normalize=True), global_step=epoch)


def eval_epoch(model, data_loader, fold, epoch):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'fold{}'.format(fold), 'eval'))

    metrics = {
        'loss': utils.Mean(),
    }

    model.eval()
    with torch.no_grad():
        fold_labels = []
        fold_logits = []
        fold_ids = []

        for images, feats, labels, ids in tqdm(data_loader, desc='epoch {} evaluation'.format(epoch)):
            images, feats, labels = images.to(DEVICE), feats.to(DEVICE), labels.to(DEVICE)
            logits = model(images, feats)

            loss = compute_loss(input=logits, target=labels)
            logits, _ = logits
            metrics['loss'].update(loss.data.cpu().numpy())

            fold_labels.append(labels)
            fold_logits.append(logits)
            fold_ids.extend(ids)

        fold_labels = torch.cat(fold_labels, 0)
        fold_logits = torch.cat(fold_logits, 0)
        assert all(data_loader.dataset.data['id_code'] == fold_ids)
        if epoch % 10 == 0:
            temp, metric, fig = find_temp_global(input=fold_logits, target=fold_labels, data=data_loader.dataset.data)
            writer.add_scalar('temp', temp, global_step=epoch)
            writer.add_scalar('metric_final', metric, global_step=epoch)
            writer.add_figure('temps', fig, global_step=epoch)
        temp = 1.  # use default temp
        fold_preds = assign_classes(
            probs=(fold_logits * temp).softmax(1).data.cpu().numpy(), data=data_loader.dataset.data)
        fold_preds = torch.tensor(fold_preds).to(fold_logits.device)
        metric = compute_metric(input=fold_preds, target=fold_labels)

        metrics = {k: metrics[k].compute_and_reset() for k in metrics}
        for k in metric:
            metrics[k] = metric[k].mean().data.cpu().numpy()
        images = images_to_rgb(images)[:16]
        print('[FOLD {}][EPOCH {}][EVAL] {}'.format(
            fold, epoch, ', '.join('{}: {:.4f}'.format(k, metrics[k]) for k in metrics)))
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

        score = metric['accuracy@1']

        scheduler.step_epoch()
        scheduler.step_score(score)

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(args.experiment_path, 'model_{}.pth'.format(fold)))


def build_submission(folds, test_data, temp):
    with torch.no_grad():
        probs = 0.

        for fold in folds:
            fold_logits, fold_ids = predict_on_test_using_fold(fold, test_data)
            fold_probs = (fold_logits * temp).softmax(2).mean(1)

            probs = probs + fold_probs
            ids = fold_ids

        probs = probs / len(folds)
        probs = probs.data.cpu().numpy()
        assert len(ids) == len(probs)
        assert all(test_data['id_code'] == ids)
        classes = assign_classes(probs=probs, data=test_data)

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
        fold_ids = []
        for images, feats, ids in tqdm(test_data_loader, desc='fold {} inference'.format(fold)):
            images, feats = images.to(DEVICE), feats.to(DEVICE)

            b, n, c, h, w = images.size()
            images = images.view(b * n, c, h, w)
            feats = feats.view(b, 1, 2).repeat(1, n, 1).view(b * n, 2)
            logits = model(images, feats)
            logits = logits.view(b, n, NUM_CLASSES)

            fold_logits.append(logits)
            fold_ids.extend(ids)

        fold_logits = torch.cat(fold_logits, 0)

    return fold_logits, fold_ids


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
        fold_ids = []

        for images, feats, labels, ids in tqdm(eval_data_loader, desc='fold {} evaluation'.format(fold)):
            images, feats, labels = images.to(DEVICE), feats.to(DEVICE), labels.to(DEVICE)
            logits = model(images, feats)

            fold_labels.append(labels)
            fold_logits.append(logits)
            fold_ids.extend(ids)

        fold_labels = torch.cat(fold_labels, 0)
        fold_logits = torch.cat(fold_logits, 0)
        assert all(eval_data['id_code'] == fold_ids)

        return fold_labels, fold_logits, fold_ids, eval_data


def find_temp_for_folds(folds, train_eval_data):
    with torch.no_grad():
        labels = []
        logits = []
        data = []
        for fold in folds:
            fold_labels, fold_logits, _, fold_data = predict_on_eval_using_fold(fold, train_eval_data)
            labels.append(fold_labels)
            logits.append(fold_logits)
            data.append(fold_data)

        # TODO: check aggregated correctly
        logits = torch.cat(logits, 0)
        labels = torch.cat(labels, 0)
        data = pd.concat(data)
        temp, metric, _ = find_temp_global(input=logits, target=labels, data=data)
        print('metric: {:.4f}, temp: {:.4f}'.format(metric, temp))

        return temp


def main():
    utils.seed_python(config.seed)
    utils.seed_torch(config.seed)

    train_eval_data = pd.read_csv(os.path.join(args.dataset_path, 'train.csv'))
    train_eval_data['root'] = os.path.join(args.dataset_path, 'train')

    test_data = pd.read_csv(os.path.join(args.dataset_path, 'test.csv'))
    test_data['root'] = os.path.join(args.dataset_path, 'test')

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

    temp = find_temp_for_folds(folds, train_eval_data)
    build_submission(folds, test_data, temp)


if __name__ == '__main__':
    main()
