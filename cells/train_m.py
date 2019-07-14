import argparse
import gc
import itertools
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
from cells.transforms import Extract, ApplyTo, RandomFlip, RandomTranspose, Resize, CenterCrop, RandomCrop, \
    ToTensor, RandomSite, SplitInSites, NormalizedColorJitter, RandomRotation
from cells.utils import images_to_rgb
from config import Config
from lr_scheduler import OneCycleScheduler
from triplet_loss import batch_hard_triplet_loss

# TODO: triplet loss
# TODO: TVN (Typical Variation Normalization)

# TODO: smaller embeddings
# TODO: margin
# TODO: batch sampler


# TODO: cyclic, clr, cawr
# TODO: triplet mining
# TODO: reduce plato
# TODO: print fold
# TODO: correct tensorboard visualization
# TODO: correct color jitter
# TODO: visualize!!!
# TODO: normalize embedding by ref
# TODO: onecycle does not platoes
# TODO: onecycle longer last stage
# TODO: speedup eval
# TODO: mix features for same class (multiple layers)
# TODO: relevant literature
# TODO: EffNet
# TODO: optimize data loading
# TODO: rmsprop
# TODO: gradacum
# TODO: resized crop
# TODO: initialize the kernel properly in order to keep approximately the same variance that the original model had.
# TODO: learn closer to negative control and further to other batches
# TODO: add feats to image
# TODO: batch/plate effects
# TODO: better split
# TODO: k shot learning
# TODO: https://github.com/recursionpharma/rxrx1-utils
# TODO: 2 images as control
# TODO: https://data.broadinstitute.org/bbbc/image_sets.html
# TODO: https://github.com/awesomedata/awesome-public-datasets#biology
# TODO: use all controls for training
# TODO: smarter split: https://www.kaggle.com/mariakesa/pcaforvisualizingbatcheffects
# TODO: different heads for different cell types
# TODO: mix sites
# TODO: concat pool
# TODO: rmsprop
# TODO: greedy assign
# TODO: gem pool
# TODO: more out layers
# TODO: sampler for triplet/mixup
# TODO: deep supervision
# TODO: parallel temp search
# TODO: hard triplet loss
# TODO: eval site selection
# TODO: check predictions/targets name
# TODO: more fc layers for arcface
# TODO: pseudo labeling
# TODO: greedy assignment
# TODO: better minimum for lr
# TODO: grad accum
# TODO: eval with tta?
# TODO: tta
# TODO: val tta (sites)
# TODO: lr schedules
# TODO: allow shuffle of plate refs within experiment
# TODO: mixup within class
# TODO: opts
# TODO: adam
# TODO: domain adaptation
# TODO: dropout
# TODO: user other images stats
# TODO: cutout
# TODO: mixup
# TODO: other cyclic (1cycle) impl
# TODO: focal
# TODO: https://www.rxrx.ai/
# TODO: batch effects
# TODO: metric learning
# TODO: context modelling notes in rxrx
# TODO: generalization notes in rxrx


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


class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, data, batch_size, drop_last):
        super().__init__(data)

        if not isinstance(batch_size, torch.utils.data.sampler._int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.data = data
        self.batch_size = batch_size
        self.drop_last = drop_last

        label_idxs = [[] for _ in range(NUM_CLASSES)]
        for idx, label in enumerate(data['sirna']):
            label_idxs[label].append(idx)

        label_idxs = [iter(idxs) for idxs in label_idxs]
        res = []
        while True:
            size = len(res)
            for idxs in label_idxs:
                res.extend(itertools.islice(idxs, 4))
            if len(res) == size:
                break
        assert data.iloc[res[:500]]['sirna'].nunique() == 125  # TODO:
        self.res = res

    def __iter__(self):
        batch = []
        for idx in self.res:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.data) // self.batch_size
        else:
            return (len(self.data) + self.batch_size - 1) // self.batch_size


class NormalizeByRefStats(object):
    def __call__(self, input):
        image, ref_stats = input['image'], input['ref_stats']

        ref_stats = ref_stats[np.random.randint(ref_stats.size(0))]
        mean, std = torch.split(ref_stats, 1, 1)
        mean, std = mean.view(mean.size(0), 1, 1), std.view(std.size(0), 1, 1)
        image = (image - mean) / std

        return {
            **input,
            'image': image
        }


class StatColorJitter(object):
    def __init__(self):
        class_to_stats = torch.load('./stats.pth')
        class_to_stats = [s.view(*s.size(), 1, 1) for s in class_to_stats]

        self.class_to_stats = class_to_stats

    # TODO: check
    def __call__(self, input):
        stats = self.class_to_stats[input['label']]
        mean, std = stats[np.random.randint(stats.size(0))]
        dim = (1, 2)

        image = input['image']
        image = (image - image.mean(dim, keepdim=True)) / image.std(dim, keepdim=True).clamp(min=1e-7)
        image = image * std + mean

        return {
            **input,
            'image': image,
        }


train_transform = T.Compose([
    ApplyTo(
        ['image'],
        T.Compose([
            RandomSite(),
            Resize(config.resize_size),
            RandomCrop(config.image_size),
            RandomFlip(),
            RandomTranspose(),
            RandomRotation(180),
            ToTensor(),
            NormalizedColorJitter(config.aug.channel_weight),
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
            CenterCrop(config.image_size),
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
            CenterCrop(config.image_size),
            SplitInSites(),
            T.Lambda(lambda xs: torch.stack([ToTensor()(x) for x in xs], 0)),
        ])),
    # NormalizeByRefStats(),
    Extract(['image', 'feat', 'id']),
])


# TODO: use pool
def find_temp_global(input, target, data):
    temps = np.logspace(np.log(0.01), np.log(1.), 20, base=np.e)
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

    return temp, fig


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


def compute_loss(input, embeddings, target):
    ce = F.cross_entropy(input=input, target=target, reduction='none')
    tl = batch_hard_triplet_loss(embeddings, target, 1.)
    assert ce.size() == tl.size()
    loss = ce * 0.9 + tl * 0.1

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
            # alpha=0.9999,#FIXME:
            momentum=optimizer.rmsprop.momentum,
            weight_decay=optimizer.weight_decay)
    else:
        raise AssertionError('invalid OPT {}'.format(optimizer.type))


# TODO: check
# TODO: fair split
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


def lr_search(train_eval_data):
    train_dataset = TrainEvalDataset(train_eval_data, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=BatchSampler(train_dataset.data, config.batch_size, drop_last=True),
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
    for param_group in optimizer.param_groups:
        param_group['lr'] = min_lr
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    model.train()
    for images, feats, labels, ids in tqdm(train_data_loader, desc='lr search'):
        images, feats, labels = images.to(DEVICE), feats.to(DEVICE), labels.to(DEVICE)
        logits, embeddings = model(images, feats, labels)

        loss = compute_loss(input=logits, embeddings=embeddings, target=labels)

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
    for images, feats, labels, ids in tqdm(data_loader, desc='epoch {} train'.format(epoch)):
        images, feats, labels = images.to(DEVICE), feats.to(DEVICE), labels.to(DEVICE)
        logits, embeddings = model(images, feats, labels)

        loss = compute_loss(input=logits, embeddings=embeddings, target=labels)
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
    }

    model.eval()
    with torch.no_grad():
        fold_labels = []
        fold_logits = []
        fold_ids = []

        for images, feats, labels, ids in tqdm(data_loader, desc='epoch {} evaluation'.format(epoch)):
            images, feats, labels = images.to(DEVICE), feats.to(DEVICE), labels.to(DEVICE)
            logits, embeddings = model(images, feats)

            loss = compute_loss(input=logits, embeddings=embeddings, target=labels)
            metrics['loss'].update(loss.data.cpu().numpy())

            fold_labels.append(labels)
            fold_logits.append(logits)
            fold_ids.extend(ids)

        fold_labels = torch.cat(fold_labels, 0)
        fold_logits = torch.cat(fold_logits, 0)
        assert all(data_loader.dataset.data['id_code'] == fold_ids)
        # temp, fig = find_temp_global(input=fold_logits, target=fold_labels, data=data_loader.dataset.data)
        temp, fig = 1., plt.figure()
        fold_preds = assign_classes(
            probs=(fold_logits * temp).softmax(1).data.cpu().numpy(), data=data_loader.dataset.data)
        fold_preds = torch.tensor(fold_preds).to(fold_logits.device)
        metric = compute_metric(input=fold_preds, target=fold_labels)

        metrics = {k: metrics[k].compute_and_reset() for k in metrics}
        for k in metric:
            metrics[k] = metric[k].mean().data.cpu().numpy()

        images = images_to_rgb(images)[:16]
        print('[EPOCH {}][EVAL] {}'.format(
            epoch, ', '.join('{}: {:.4f}'.format(k, metrics[k]) for k in metrics)))
        for k in metrics:
            writer.add_scalar(k, metrics[k], global_step=epoch)
        writer.add_image('images', torchvision.utils.make_grid(
            images, nrow=math.ceil(math.sqrt(images.size(0))), normalize=True), global_step=epoch)
        writer.add_figure('temps', fig, global_step=epoch)

        return metrics


def train_fold(fold, train_eval_data):
    train_indices, eval_indices = indices_for_fold(fold, train_eval_data)

    train_dataset = TrainEvalDataset(train_eval_data.iloc[train_indices], transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=BatchSampler(train_dataset.data, config.batch_size, drop_last=True),
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
            feats = feats.view(b, 1).repeat(1, n).view(b * n)  # TODO: check
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


# TODO: log metric
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
        temp, fig = find_temp_global(input=logits, target=labels, data=data)

        print('temp: {:.4f}'.format(temp))

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
