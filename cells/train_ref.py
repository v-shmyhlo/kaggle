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
from PIL import Image
from tensorboardX import SummaryWriter
from tqdm import tqdm

import lr_scheduler_wrapper
import optim
import utils
from cells.dataset_ref import NUM_CLASSES, TrainEvalDataset, TestDataset
from cells.model_ref import Model
from cells.transforms import Extract, RandomFlip, RandomTranspose, Resize, ToTensor, RandomSite, SplitInSites, \
    RandomCrop, CenterCrop, NormalizeByExperimentStats, NormalizeByPlateStats, ChannelReweight
from cells.utils import images_to_rgb
from config import Config
from lr_scheduler import OneCycleScheduler
from transforms import ApplyTo

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


class RandomResize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, input):
        size = round(np.random.uniform(self.min_size, self.max_size))
        input = Resize(size)(input)

        return input


class MixSites(object):
    def __call__(self, input):
        w, h = input[0].size
        s1, s2 = SplitInSites()(input)
        if np.random.rand() > 0.5:
            s1, s2 = s2, s1

        lam = np.random.uniform(0, 1)
        r_x = np.random.uniform(0, w)
        r_y = np.random.uniform(0, h)
        r_w = w * np.sqrt(1 - lam)
        r_h = h * np.sqrt(1 - lam)
        x1 = (r_x - r_w / 2).clip(0, w).round().astype(np.int32)
        x2 = (r_x + r_w / 2).clip(0, w).round().astype(np.int32)
        y1 = (r_y - r_h / 2).clip(0, h).round().astype(np.int32)
        y2 = (r_y + r_h / 2).clip(0, h).round().astype(np.int32)

        mode = s1[0].mode
        for c in input:
            assert c.mode == mode

        s1 = [np.array(c) for c in s1]
        s2 = [np.array(c) for c in s2]

        for c1, c2 in zip(s1, s2):
            c1[x1:x2, y1:y2] = c2[x1:x2, y1:y2]

        s1 = [Image.fromarray(c) for c in s1]

        assert s1[0].mode == mode

        return s1


random_resize = Resetable(RandomResize)
random_crop = Resetable(RandomCrop)
center_crop = Resetable(CenterCrop)
to_tensor = ToTensor()

if config.normalize is None:
    normalize = T.Compose([])
elif config.normalize == 'experiment':
    normalize = NormalizeByExperimentStats(
        torch.load('./experiment_stats.pth'))  # TODO: needs realtime computation on private
elif config.normalize == 'plate':
    normalize = NormalizeByPlateStats(
        torch.load('./plate_stats.pth'))  # TODO: needs realtime computation on private
else:
    raise AssertionError('invalide normalization {}'.format(config.normalize))

train_transform = T.Compose([
    ApplyTo(
        ['image', 'ref'],
        T.Compose([
            RandomSite(),
            Resize(config.resize_size),
            random_crop,
            RandomFlip(),
            RandomTranspose(),
            to_tensor,
            ChannelReweight(config.aug.channel_weight),
        ])),
    normalize,
    Extract(['image', 'ref', 'feat', 'exp', 'label', 'id']),
])
eval_transform = T.Compose([
    ApplyTo(
        ['image', 'ref'],
        T.Compose([
            RandomSite(),  # FIXME:
            Resize(config.resize_size),
            center_crop,
            to_tensor,
        ])),
    normalize,
    Extract(['image', 'ref', 'feat', 'exp', 'label', 'id']),
])
test_transform = T.Compose([
    ApplyTo(
        ['image', 'ref'],
        T.Compose([
            Resize(config.resize_size),
            center_crop,
            SplitInSites(),
            T.Lambda(lambda xs: torch.stack([to_tensor(x) for x in xs], 0)),
        ])),
    normalize,
    Extract(['image', 'ref', 'feat', 'exp', 'id']),
])


def update_transforms(p):
    assert 0. <= p <= 1.

    crop_size = round(224 + (config.crop_size - 224) * p)
    delta = config.resize_size - crop_size
    resize_size = config.resize_size - delta, config.resize_size + delta
    assert sum(resize_size) / 2 == config.resize_size
    print('update transforms p: {:.2f}, resize_size: {}, crop_size: {}'.format(p, resize_size, crop_size))

    random_resize.reset(*resize_size)
    random_crop.reset(crop_size)
    center_crop.reset(crop_size)


# TODO: use pool
def find_temp_global(input, target, exps):
    temps = np.logspace(np.log(1e-4), np.log(1.0), 50, base=np.e)
    metrics = []
    for temp in tqdm(temps, desc='temp search'):
        fold_preds = assign_classes(probs=(input * temp).softmax(1).data.cpu().numpy(), exps=exps)
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


def compute_loss(input, target):
    loss = F.cross_entropy(input=input, target=target, reduction='none')

    return loss


def compute_metric(input, target):
    metric = {
        'accuracy@1': (input == target).float(),
    }

    return metric


def assign_classes(probs, exps):
    # TODO: refactor numpy/torch usage

    exps = np.array(exps)
    classes = np.zeros(probs.shape[0], dtype=np.int64)
    for exp in np.unique(exps):
        subset = exps == exp
        preds = probs[subset]
        _, c, _ = lap.lapjv(1 - preds, extend_cost=True)
        classes[subset] = c

    return classes


def build_optimizer(optimizer_config, parameters):
    if optimizer_config.type == 'sgd':
        optimizer = torch.optim.SGD(
            parameters,
            optimizer_config.lr,
            momentum=optimizer_config.sgd.momentum,
            weight_decay=optimizer_config.weight_decay,
            nesterov=True)
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

    update_transforms(1.)
    model.train()
    optimizer.zero_grad()
    for i, (images, refs, feats, _, labels, _) in enumerate(tqdm(train_eval_data_loader, desc='lr search'), 1):
        images, refs, feats, labels = images.to(DEVICE), refs.to(DEVICE), feats.to(DEVICE), labels.to(DEVICE)
        logits = model(images, refs, feats, labels)

        loss = compute_loss(input=logits, target=labels)

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
    for i, (images, refs, feats, _, labels, _) in enumerate(tqdm(data_loader, desc='epoch {} train'.format(epoch)), 1):
        images, refs, feats, labels = images.to(DEVICE), refs.to(DEVICE), feats.to(DEVICE), labels.to(DEVICE)
        logits = model(images, refs, feats, labels)

        loss = compute_loss(input=logits, target=labels)
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
        fold_exps = []

        for images, refs, feats, exps, labels, _ in tqdm(data_loader, desc='epoch {} evaluation'.format(epoch)):
            images, refs, feats, labels = images.to(DEVICE), refs.to(DEVICE), feats.to(DEVICE), labels.to(DEVICE)
            logits = model(images, refs, feats)

            loss = compute_loss(input=logits, target=labels)
            metrics['loss'].update(loss.data.cpu().numpy())

            fold_labels.append(labels)
            fold_logits.append(logits)
            fold_exps.extend(exps)

        fold_labels = torch.cat(fold_labels, 0)
        fold_logits = torch.cat(fold_logits, 0)

        if epoch % 10 == 0:
            temp, metric, fig = find_temp_global(input=fold_logits, target=fold_labels, exps=fold_exps)
            writer.add_scalar('temp', temp, global_step=epoch)
            writer.add_scalar('metric_final', metric, global_step=epoch)
            writer.add_figure('temps', fig, global_step=epoch)
        temp = 1.  # use default temp
        fold_preds = assign_classes(probs=(fold_logits * temp).softmax(1).data.cpu().numpy(), exps=fold_exps)
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
                beta_range=config.sched.onecycle.beta,
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
    eval_dataset = TrainEvalDataset(train_eval_data.iloc[eval_indices], transform=eval_transform)
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

    update_transforms(1.)  # FIXME:
    temp = find_temp_for_folds(folds, train_eval_data)
    gc.collect()
    build_submission(folds, test_data, temp)


if __name__ == '__main__':
    main()
