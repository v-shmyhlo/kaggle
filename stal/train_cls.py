import argparse
import gc
import math
import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributions
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as T
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tensorboardX import SummaryWriter
from tqdm import tqdm

import lr_scheduler_wrapper
import optim
import utils
from config import Config
from losses import softmax_cross_entropy, softmax_focal_loss, sigmoid_focal_loss, dice_loss
from lr_scheduler import OneCycleScheduler
from radam import RAdam
from stal.compute_image_stats import compute_buckets
from stal.dataset import NUM_CLASSES, TrainEvalDataset, TestDataset, build_data
from stal.model_cls import Model, Ensemble
from stal.transforms import RandomHorizontalFlip, RandomVerticalFlip, SampledRandomCrop, RandomCrop
from stal.utils import mask_to_image, rle_encode
from transforms import ApplyTo, Extract, Resetable, RandomGamma, RandomBrightness, RandomContrast

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
config = Config.from_json(args.config_path)
shutil.copy(args.config_path, os.path.join(utils.mkdir(args.experiment_path), 'config.yaml'))

normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

if config.aug.crop.type == 'random':
    random_crop = Resetable(RandomCrop)
elif config.aug.crop.type == 'sampled':
    random_crop = Resetable(SampledRandomCrop)
else:
    raise AssertionError('invalid config.aug.crop.type {}'.format(config.aug.crop.type))

train_transform = T.Compose([
    random_crop,
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    ApplyTo(
        ['image'],
        T.Compose([
            RandomBrightness(config.aug.brightness),
            RandomContrast(config.aug.contrast),
            RandomGamma(config.aug.gamma),
            T.ToTensor(),
            normalize,
        ])),
    ApplyTo(
        ['mask'],
        T.Compose([
            T.ToTensor(),
        ])),
    Extract(['image', 'mask', 'id']),
])
eval_transform = T.Compose([
    ApplyTo(
        ['image'],
        T.Compose([
            T.ToTensor(),
            normalize,
        ])),
    ApplyTo(
        ['mask'],
        T.Compose([
            T.ToTensor(),
        ])),
    Extract(['image', 'mask', 'id']),
])
test_transform = T.Compose([
    ApplyTo(
        ['image'],
        T.Compose([
            T.ToTensor(),
            normalize,
            T.Lambda(lambda x: torch.stack([x], 0))
        ])),
    Extract(['image', 'id']),
])


def cut_mix(images_1, labels_1):
    b, _, h, w = images_1.size()
    perm = np.random.permutation(b)
    images_2, labels_2 = images_1[perm], labels_1[perm]

    lam = np.random.uniform(0, 1)
    r_x = np.random.uniform(0, w)
    r_y = np.random.uniform(0, h)
    r_w = w * np.sqrt(1 - lam)
    r_h = h * np.sqrt(1 - lam)
    x1 = (r_x - r_w / 2).clip(0, w).round().astype(np.int32)
    x2 = (r_x + r_w / 2).clip(0, w).round().astype(np.int32)
    y1 = (r_y - r_h / 2).clip(0, h).round().astype(np.int32)
    y2 = (r_y + r_h / 2).clip(0, h).round().astype(np.int32)

    images_1[:, :, x1:x2, y1:y2] = images_2[:, :, x1:x2, y1:y2]
    labels_1[:, :, x1:x2, y1:y2] = labels_2[:, :, x1:x2, y1:y2]

    images = images_1
    labels = labels_1

    return images, labels


def update_transforms(p):
    assert 0. <= p <= 1.

    crop_size = round(config.aug.crop.size.min + (config.aug.crop.size.max - config.aug.crop.size.min) * p)
    crop_size = round(crop_size / 2**5) * 2**5
    print('update transforms p: {:.2f}, crop_size: {}'.format(p, crop_size))
    random_crop.reset((256, crop_size))
    # center_crop.reset(crop_size)


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


def one_hot(input):
    return utils.one_hot(input, num_classes=NUM_CLASSES).permute((0, 3, 1, 2))


def compute_loss(class_input, mask_input, target):
    class_loss = compute_class_loss(input=class_input, target=target)
    mask_loss = compute_mask_loss(input=mask_input, target=target)

    # assert class_loss.size() == mask_loss.size()
    loss = class_loss.mean() + mask_loss.mean()

    return loss


def compute_class_loss(input, target, axis=(2, 3)):
    input = input[:, 1:]
    target = target.sum(axis)[:, 1:]
    target = (target > 0.).float()

    # ce = sigmoid_cross_entropy(input=input, target=target).mean(1)
    focal = sigmoid_focal_loss(input=input, target=target).mean(1)

    loss = [
        # ce,
        focal,
    ]
    assert all(l.size() == loss[0].size() for l in loss)
    loss = sum(loss) / len(loss)

    return loss


def compute_mask_loss(input, target, axis=(2, 3)):
    # ce = softmax_cross_entropy(input=input, target=target, axis=1, keepdim=True).mean(axis).mean(1)
    focal = softmax_focal_loss(input=input, target=target, axis=1, keepdim=True).mean(axis).mean(1)
    dice = dice_loss(input=input.softmax(1), target=target, axis=axis).mean(1)
    # tversky = tversky_loss(input=input.softmax(1), target=target, alpha=0.1, beta=0.9, axis=axis).mean(1)

    loss = [
        # ce ,
        focal,
        dice,
        # tversky,
    ]
    assert all(l.size() == loss[0].size() for l in loss)
    loss = sum(loss) / len(loss)

    mask = target[:, 1:].sum(axis).sum(1) > 0.
    assert mask.shape == loss.shape
    loss = loss[mask]

    return loss


# TODO: check correctness
# TODO: use argmax after masking
def compute_metric(class_input, mask_input, target, axis=(2, 3)):
    class_pred = (class_input[:, 1:] > 0.).float()
    del class_input

    mask_pred = one_hot(mask_input.argmax(1))[:, 1:]
    del mask_input

    target = target[:, 1:]
    mask_pred = mask_pred * class_pred.view(*class_pred.size(), 1, 1)

    intersection = (mask_pred * target).sum(axis)
    union = mask_pred.sum(axis) + target.sum(axis)
    dice = (2. * intersection) / union
    dice[union == 0.] = 1.

    target = (target.sum(axis) > 0.).float()
    class_eq = (class_pred == target).float()

    metric = {
        'dice': dice,
        'class_acc/0': class_eq[:, 0],
        'class_acc/1': class_eq[:, 1],
        'class_acc/2': class_eq[:, 2],
        'class_acc/3': class_eq[:, 3],
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
    elif optimizer_config.type == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            parameters,
            optimizer_config.lr,
            momentum=optimizer_config.rmsprop.momentum,
            weight_decay=optimizer_config.weight_decay)
    elif optimizer_config.type == 'adam':
        optimizer = torch.optim.Adam(
            parameters,
            optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay)
    elif optimizer_config.type == 'radam':
        optimizer = RAdam(
            parameters,
            optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay)
    else:
        raise AssertionError('invalid OPT {}'.format(optimizer_config.type))

    if optimizer_config.lookahead is not None:
        optimizer = optim.LA(
            optimizer,
            optimizer_config.lookahead.lr,
            num_steps=optimizer_config.lookahead.steps)

    if optimizer_config.ewa is not None:
        optimizer = optim.EWA(
            optimizer,
            optimizer_config.ewa.momentum,
            num_steps=optimizer_config.ewa.steps)
    else:
        optimizer = optim.DummySwitchable(optimizer)

    return optimizer


def indices_for_fold(fold, data):
    areas = np.load('./stal/stats.npy')
    num_buckets = areas.shape[0] // 300
    buckets = compute_buckets(areas, num_buckets=num_buckets)
    print(buckets.shape)
    buckets = np.eye(num_buckets + 1)[buckets]
    print(buckets.shape)
    buckets = buckets.reshape(buckets.shape[0], buckets.shape[1] * buckets.shape[2])
    print(buckets.shape)

    kfold = MultilabelStratifiedKFold(len(FOLDS), shuffle=True, random_state=config.seed)
    splits = list(kfold.split(np.zeros(len(data)), buckets))
    train_indices, eval_indices = splits[fold - 1]
    assert len(train_indices) + len(eval_indices) == len(data)

    for c in range(areas.shape[1]):
        for i in [train_indices, eval_indices]:
            a = areas[i, c]
            print('class: {}, mean: {:.2f}, std: {:.2f}, min: {:.2f}, max: {:.2f}'.format(
                c, a.mean(), a.std(), a.min(), a.max()))

    print(len(train_indices) / len(eval_indices))

    weights = np.log(np.e + areas.mean(1))

    # mask = (areas > 0.).astype(np.float)
    # weights = (1 - mask).sum(0, keepdims=True) / mask.sum(0, keepdims=True)
    # weights = mask * weights + (1 - mask)
    # print((mask * weights).sum(0) / ((1 - mask) * weights).sum(0))
    # weights = weights.mean(1)

    weights = weights / weights.sum()
    print(weights.min(), weights.max(), weights.max() / weights.min())

    return train_indices, eval_indices, weights


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
        class_logits, mask_logits = model(images)

        loss = compute_loss(class_input=class_logits, mask_input=mask_logits, target=masks)

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
        'fps': utils.Mean(),
    }

    update_transforms(np.linspace(0, 1, config.epochs)[epoch - 1].item())
    model.train()
    optimizer.zero_grad()
    t1 = time.time()
    for i, (images, masks, ids) in enumerate(tqdm(data_loader, desc='epoch {} train'.format(epoch)), 1):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        # images, masks = cut_mix(images, masks)

        class_logits, mask_logits = model(images)

        loss = compute_loss(class_input=class_logits, mask_input=mask_logits, target=masks)
        metrics['loss'].update(loss.data.cpu().numpy())

        lr = scheduler.get_lr()
        (loss.mean() / config.opt.acc_steps).backward()

        # with amp.scale_loss((loss.mean() / config.opt.acc_steps), optimizer) as scaled_loss:
        #     scaled_loss.backward()

        if i % config.opt.acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        t2 = time.time()
        metrics['fps'].update(1 / ((t2 - t1) / images.size(0)))
        t1 = t2

    with torch.no_grad():
        metrics = {k: metrics[k].compute_and_reset() for k in metrics}
        print('[FOLD {}][EPOCH {}][TRAIN] {}'.format(
            fold, epoch, ', '.join('{}: {:.4f}'.format(k, metrics[k]) for k in metrics)))
        for k in metrics:
            writer.add_scalar(k, metrics[k], global_step=epoch)
        writer.add_scalar('learning_rate', lr, global_step=epoch)

        images = images[:32]
        masks = mask_to_image(masks[:32], num_classes=NUM_CLASSES)
        preds = mask_to_image(one_hot(mask_logits[:32].argmax(1)), num_classes=NUM_CLASSES)
        probs = mask_to_image(mask_logits[:32].softmax(1), num_classes=NUM_CLASSES)

        writer.add_image('images', torchvision.utils.make_grid(
            images, nrow=compute_nrow(images), normalize=True), global_step=epoch)
        writer.add_image('masks', torchvision.utils.make_grid(
            masks, nrow=compute_nrow(masks), normalize=True), global_step=epoch)
        writer.add_image('preds', torchvision.utils.make_grid(
            preds, nrow=compute_nrow(preds), normalize=True), global_step=epoch)
        writer.add_image('probs', torchvision.utils.make_grid(
            probs, nrow=compute_nrow(probs), normalize=True), global_step=epoch)


def eval_epoch(model, data_loader, fold, epoch):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'fold{}'.format(fold), 'eval'))

    metrics = {
        'loss': utils.Mean(),
        'dice': utils.Mean(),
        'fps': utils.Mean(),
        'entropy': utils.Mean(),

        'class_acc/0': utils.Mean(),
        'class_acc/1': utils.Mean(),
        'class_acc/2': utils.Mean(),
        'class_acc/3': utils.Mean(),
    }

    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for images, masks, _ in tqdm(data_loader, desc='epoch {} evaluation'.format(epoch)):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            class_logits, mask_logits = model(images)

            loss = compute_loss(class_input=class_logits, mask_input=mask_logits, target=masks)
            metrics['loss'].update(loss.data.cpu().numpy())

            metric = compute_metric(class_input=class_logits, mask_input=mask_logits, target=masks)
            for k in metric:
                metrics[k].update(metric[k].data.cpu().numpy())

            ent = softmax_cross_entropy(input=mask_logits, target=mask_logits.softmax(1), axis=1).mean((1, 2))
            metrics['entropy'].update(ent.data.cpu().numpy())

            t2 = time.time()
            metrics['fps'].update(1 / ((t2 - t1) / images.size(0)))
            t1 = t2

        metrics = {k: metrics[k].compute_and_reset() for k in metrics}
        print('[FOLD {}][EPOCH {}][EVAL] {}'.format(
            fold, epoch, ', '.join('{}: {:.4f}'.format(k, metrics[k]) for k in metrics)))
        for k in metrics:
            writer.add_scalar(k, metrics[k], global_step=epoch)

        images = images[:32]
        masks = mask_to_image(masks[:32], num_classes=NUM_CLASSES)
        preds = mask_to_image(one_hot(mask_logits[:32].argmax(1)), num_classes=NUM_CLASSES)
        probs = mask_to_image(mask_logits[:32].softmax(1), num_classes=NUM_CLASSES)

        writer.add_image('images', torchvision.utils.make_grid(
            images, nrow=compute_nrow(images), normalize=True), global_step=epoch)
        writer.add_image('masks', torchvision.utils.make_grid(
            masks, nrow=compute_nrow(masks), normalize=True), global_step=epoch)
        writer.add_image('preds', torchvision.utils.make_grid(
            preds, nrow=compute_nrow(preds), normalize=True), global_step=epoch)
        writer.add_image('probs', torchvision.utils.make_grid(
            probs, nrow=compute_nrow(probs), normalize=True), global_step=epoch)

        return metrics


def train_fold(fold, train_eval_data, tmp):
    train_indices, eval_indices, _ = indices_for_fold(fold, train_eval_data)  # FIXME: dataset size

    train_data = train_eval_data.iloc[train_indices]
    eval_data = train_eval_data.iloc[eval_indices]

    # train_data = pd.concat([
    #     train_data,
    #     tmp,
    # ])

    train_dataset = TrainEvalDataset(train_data, transform=train_transform)
    # weights = weights[train_indices]
    # assert len(train_dataset) == len(weights)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        drop_last=True,
        # sampler=torch.utils.data.WeightedRandomSampler(weights, len(train_dataset), replacement=True),
        shuffle=True,
        num_workers=args.workers,
        worker_init_fn=worker_init_fn)
    eval_dataset = TrainEvalDataset(eval_data, transform=eval_transform)
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

    # model, optimizer = amp.initialize(model, optimizer, opt_level='O0')

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
    elif config.sched.type == 'cosine':
        scheduler = lr_scheduler_wrapper.StepWrapper(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=len(train_data_loader) * config.epochs))
    elif config.sched.type == 'step':
        scheduler = lr_scheduler_wrapper.EpochWrapper(
            torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.sched.step.step_size,
                gamma=config.sched.step.decay))
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
                mode='triangular2',
                gamma=config.sched.cyclic.decay**(1 / (step_size_up + step_size_down)),
                cycle_momentum=True,
                base_momentum=0.85,
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
        optimizer.train()
        train_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data_loader=train_data_loader,
            fold=fold,
            epoch=epoch)
        gc.collect()
        optimizer.eval()
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


# def build_submission(folds, test_data, temp):
#     with torch.no_grad():
#         probs = 0.
#
#         for fold in folds:
#             fold_logits, fold_exps, fold_ids = predict_on_test_using_fold(fold, test_data)
#             fold_probs = (fold_logits * temp).softmax(2).mean(1)
#
#             probs = probs + fold_probs
#             exps = fold_exps
#             ids = fold_ids
#
#         probs = probs / len(folds)
#         probs = probs.data.cpu().numpy()
#         assert len(probs) == len(exps) == len(ids)
#         classes = assign_classes(probs=probs, exps=exps)
#
#         submission = pd.DataFrame({'id_code': ids, 'sirna': classes})
#         submission.to_csv(os.path.join(args.experiment_path, 'submission.csv'), index=False)
#         submission.to_csv('./submission.csv', index=False)


def build_submission(folds, test_data, temp):
    with torch.no_grad():
        rles, ids = predict_on_test_using_fold(folds, test_data)

        paths = [
            ('utils.py', 'utils.py'),
            ('transforms.py', 'transforms.py'),
            ('config.py', 'config.py'),
            ('stal/infer.py', 'stal/infer.py'),
            ('stal/model.py', 'stal/model.py'),
            ('stal/model_cls.py', 'stal/model_cls.py'),
            ('stal/transforms.py', 'stal/transforms.py'),
            ('stal/dataset.py', 'stal/dataset.py'),
            ('stal/utils.py', 'stal/utils.py'),
            (os.path.join(args.experiment_path, 'config.yaml'), 'experiment/config.yaml')
        ]
        for fold in folds:
            paths.append((
                os.path.join(args.experiment_path, 'model_{}.pth'.format(fold)),
                'experiment/model_{}.pth'.format(fold)))
        utils.mkdir('subs/stal')
        utils.mkdir('subs/experiment')
        for src, dst in paths:
            shutil.copy(src, os.path.join('subs', dst))


def predict_on_test_using_fold(folds, test_data):
    test_dataset = TestDataset(test_data, transform=test_transform)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=args.workers,
        worker_init_fn=worker_init_fn)

    models = []
    for fold in folds:
        model = Model(config.model, NUM_CLASSES, pretrained=False)
        model = model.to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(args.experiment_path, 'model_{}.pth'.format(fold))))
        models.append(model)
    model = Ensemble(models)
    del fold

    model.eval()
    with torch.no_grad():
        fold_rles = []
        fold_ids = []

        for images, ids in tqdm(test_data_loader, desc='inference'):
            images = images.to(DEVICE)

            b, nt, c, h, w = images.size()
            images = images.view(b * nt, c, h, w)
            class_logits, mask_logits = model(images)
            _, nm, _ = class_logits.size()
            class_logits = class_logits.view(b, nt * nm, NUM_CLASSES)
            mask_logits = mask_logits.view(b, nt * nm, NUM_CLASSES, h, w)

            n_dim = 1
            c_dim = 2

            class_probs = class_logits.sigmoid().mean(n_dim)
            mask_probs = mask_logits.softmax(c_dim).mean(n_dim)

            tmp = mask_probs.clone()
            tmp[:, 1:] *= (class_probs[:, 1:] > 0.5).float().view(class_probs.size(0), class_probs.size(1) - 1, 1, 1)
            tmp /= tmp.sum(1, keepdim=True)
            assert torch.allclose(tmp.sum(1), torch.ones(tmp.size(0), tmp.size(2), tmp.size(3), device=tmp.device))

            # TODO: fixme
            for id, m in zip(ids, tmp.permute(0, 2, 3, 1).data.cpu().numpy()):
                np.save('./pl/{}.npy'.format(id), m)
            continue

            class_probs = class_probs[:, 1:]
            mask_probs = one_hot(mask_probs.argmax(1))[:, 1:]

            class_probs = (class_probs > 0.5).float().view(class_probs.size(0), class_probs.size(1), 1, 1)
            mask_probs = mask_probs * class_probs

            rles = [
                [rle_encode(c) for c in mask]
                for mask in mask_probs.data.cpu().numpy()
            ]

            fold_rles.extend(rles)
            fold_ids.extend(ids)

    return fold_rles, fold_ids


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
    train_eval_data = build_data(train_eval_data)

    tmp = pd.read_csv(os.path.join(args.dataset_path, 'sample_submission.csv'), converters={'EncodedPixels': str})
    tmp['root'] = os.path.join(args.dataset_path, 'test_images')
    tmp = build_data(tmp)
    tmp['rles'] = None

    test_data = pd.read_csv(os.path.join(args.dataset_path, 'sample_submission.csv'), converters={'EncodedPixels': str})
    test_data['root'] = os.path.join(args.dataset_path, 'test_images')
    test_data = build_data(test_data)

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
            train_fold(fold, train_eval_data, tmp)

    update_transforms(1.)  # FIXME:
    # temp = find_temp_for_folds(folds, train_eval_data)
    temp = None
    gc.collect()
    build_submission(folds, test_data, temp)


if __name__ == '__main__':
    main()
