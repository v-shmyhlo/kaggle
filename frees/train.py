import numpy as np
import gc
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image
import argparse
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold
from lr_scheduler import OneCycleScheduler
from optim import AdamW
import utils
from augmentation import SquarePad
from .model import Model
from .dataset import NUM_CLASSES, TrainEvalDataset, TestDataset

# TODO: sgd
# TODO: check del
# TODO: try largest lr before diverging
# TODO: check all plots rendered
# TODO: adamw

FOLDS = list(range(1, 5 + 1))

parser = argparse.ArgumentParser()
parser.add_argument('--experiment-path', type=str, default='./tf_log/frees')
parser.add_argument('--dataset-path', type=str, required=True)
parser.add_argument('--workers', type=int, default=os.cpu_count())
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--fold', type=int, choices=FOLDS)
parser.add_argument('--image-size', type=int, default=128)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--beta', type=float, nargs=2, default=(0.95, 0.85))
parser.add_argument('--anneal', type=str, choices=['linear', 'cosine'], default='linear')
# parser.add_argument('--aug', type=str, choices=['low', 'med', 'med+color', 'hard', 'pad', 'pad-hard'], default='med')
parser.add_argument('--opt', type=str, choices=['adam', 'adamw', 'sgd'], default='adam')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

utils.seed_everything(args.seed)


# train_data = pd.read_csv(os.path.join(args.dataset_path, 'train.csv'))
# train_data['attribute_ids'] = train_data['attribute_ids'].apply(lambda s: [int(x) for x in s.split()])
#
# classes = pd.read_csv(os.path.join(args.dataset_path, 'labels.csv'))
# id_to_class = {row['attribute_id']: row['attribute_name'] for _, row in classes.iterrows()}
# class_to_id = {id_to_class[k]: k for k in id_to_class}


def f2_loss(input, target, eps=1e-7):
    input = input.sigmoid()

    tp = (target * input).sum(1)
    tn = ((1 - target) * (1 - input)).sum(1)
    fp = ((1 - target) * input).sum(1)
    fn = (target * (1 - input)).sum(1)

    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)

    beta_sq = 2**2
    f2 = (1 + beta_sq) * p * r / (beta_sq * p + r + eps)
    loss = -(f2 + eps).log()

    return loss


def hinge_loss(input, target, delta=1.):
    target = target * 1. + (1 - target) * -1.
    loss = torch.max(torch.zeros_like(input).to(input.device), delta - target * input)

    return loss


def compute_loss(input, target):
    loss = [l(input=input, target=target).mean() for l in LOSS]
    loss = sum(loss) / len(loss)

    return loss


def compute_score(input, target, threshold=0.5):
    input = (input.sigmoid() > threshold).float()

    tp = (target * input).sum(-1)
    tn = ((1 - target) * (1 - input)).sum(-1)
    fp = ((1 - target) * input).sum(-1)
    fn = (target * (1 - input)).sum(-1)

    p = tp / (tp + fp)
    r = tp / (tp + fn)

    beta_sq = 2**2
    f2 = (1 + beta_sq) * p * r / (beta_sq * p + r)
    f2[f2 != f2] = 0.

    return f2


def find_threshold_global(input, target):
    thresholds = np.arange(0.1, 0.9, 0.01)
    scores = [compute_score(input=input, target=target, threshold=t).mean()
              for t in tqdm(thresholds, desc='threshold search')]
    threshold = thresholds[np.argmax(scores)]
    score = scores[np.argmax(scores)]

    plt.plot(thresholds, scores)
    plt.axvline(threshold)
    plt.title('score: {:.4f}, threshold: {:.4f}'.format(score.item(), threshold))
    plot = utils.plot_to_image()

    return threshold, score, plot


# NUM_CLASSES = len(classes)
# ARCH = 'seresnext50'
# LOSS_SMOOTHING = 0.9
# LOSS = [
#     # f2_loss,
#     # F.binary_cross_entropy_with_logits,
#     FocalLoss(),
#     # hinge_loss,
# ]
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# TODO: pin memory
# TODO: stochastic weight averaging
# TODO: group images by buckets (size, ratio) and batch
# TODO: hinge loss clamp instead of minimum
# TODO: losses
# TODO: better one cycle
# TODO: cos vs lin
# TODO: load and restore state after lr finder
# TODO: better loss smoothing
# TODO: shuffle thresh search
# TODO: init thresh search from global best
# TODO: shuffle split
# TODO: tune on large size
# TODO: cross val
# TODO: smart sampling
# TODO: larger model
# TODO: imagenet papers
# TODO: load image as jpeg
# TODO: augmentations (flip, crops, color)
# TODO: min 1 tag?
# TODO: pick threshold to match ratio
# TODO: compute smoothing beta from batch size and num steps
# TODO: speedup image loading
# TODO: pin memory
# TODO: smart sampling
# TODO: better threshold search (step, epochs)
# TODO: weight standartization
# TODO: label smoothing
# TODO: build sched for lr find

class RandomCrop(object):
    def __init__(self, size):
        pass

    def __call__(self, input):
        print(input.shape)
        fial


train_transform = T.Compose([
    RandomCrop(256)

])
eval_transform = None
test_transform = None


# TODO: should use top momentum to pick best lr?
def build_optimizer(optimizer, parameters, lr, beta, weight_decay):
    if optimizer == 'adam':
        return torch.optim.Adam(parameters, lr, betas=(beta, 0.999), weight_decay=weight_decay)
    elif optimizer == 'adamw':
        return AdamW(parameters, lr, betas=(beta, 0.999), weight_decay=weight_decay)
    elif optimizer == 'sgd':
        return torch.optim.SGD(parameters, lr, momentum=beta, weight_decay=weight_decay)
    else:
        raise AssertionError('invalid OPT {}'.format(optimizer))


def indices_for_fold(fold, dataset_size):
    kfold = KFold(len(FOLDS), shuffle=True, random_state=args.seed)
    splits = list(kfold.split(np.zeros(dataset_size)))
    train_indices, eval_indices = splits[fold - 1]
    assert len(train_indices) + len(eval_indices) == dataset_size

    return train_indices, eval_indices


def find_lr(train_data):
    train_dataset = TrainEvalDataset(train_data, transform=train_transform)
    # TODO: all args
    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=args.workers)

    min_lr = 1e-8
    max_lr = 10.
    gamma = (max_lr / min_lr)**(1 / len(data_loader))

    lrs = []
    losses = []
    lim = None

    model = Model(NUM_CLASSES)
    model = model.to(DEVICE)
    optimizer = build_optimizer(args.opt, model.parameters(), min_lr, args.beta[-1], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    model.train()
    for images, labels, ids in tqdm(data_loader, desc='lr search'):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)

        loss = compute_loss(input=logits, target=labels)

        lrs.append(np.squeeze(scheduler.get_lr()))
        losses.append(loss.data.cpu().numpy().mean())

        if lim is None:
            lim = losses[0] * 1.1

        if lim < losses[-1]:
            break

        scheduler.step()
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        if args.debug:
            break

    with torch.no_grad():
        losses = np.clip(losses, 0, lim)
        minima = {
            'loss': losses[np.argmin(utils.smooth(losses))],
            'lr': lrs[np.argmin(utils.smooth(losses))]
        }

        writer = SummaryWriter(os.path.join(args.experiment_path, 'lr_search'))

        step = 0
        for loss, loss_sm in zip(losses, utils.smooth(losses)):
            writer.add_scalar('search_loss', loss, global_step=step)
            writer.add_scalar('search_loss_sm', loss_sm, global_step=step)
            step += args.batch_size

        np.save('stats.npy', (lrs, losses))

        plt.plot(lrs, losses)
        plt.plot(lrs, utils.smooth(losses))
        plt.axvline(minima['lr'])
        plt.xscale('log')
        plt.title('loss: {:.8f}, lr: {:.8f}'.format(minima['loss'], minima['lr']))
        plot = utils.plot_to_image()
        writer.add_image('search', plot.transpose((2, 0, 1)), global_step=0)

        return minima


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

        scheduler.step()
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        if args.debug:
            break

    with torch.no_grad():
        loss = metrics['loss'].compute_and_reset()

        print('[FOLD {}][EPOCH {}][TRAIN] loss: {:.4f}'.format(fold, epoch, loss))
        writer.add_scalar('loss', loss, global_step=epoch)
        lr, beta = scheduler.get_lr()
        writer.add_scalar('learning_rate', lr, global_step=epoch)
        writer.add_scalar('beta', beta, global_step=epoch)
        writer.add_image('image', torchvision.utils.make_grid(images[:32], normalize=True), global_step=epoch)


def eval_epoch(model, data_loader, fold, epoch):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'fold{}'.format(fold), 'eval'))

    metrics = {
        'loss': utils.Mean(),
    }

    predictions = []
    targets = []
    model.eval()
    with torch.no_grad():
        for images, labels, ids in tqdm(data_loader, desc='epoch {} evaluation'.format(epoch)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)

            targets.append(labels)
            predictions.append(logits)

            loss = compute_loss(input=logits, target=labels)
            metrics['loss'].update(loss.data.cpu().numpy())

            if args.debug:
                break

        loss = metrics['loss'].compute_and_reset()

        predictions = torch.cat(predictions, 0)
        targets = torch.cat(targets, 0)
        threshold, score, plot = find_threshold_global(input=predictions, target=targets)

        print('[FOLD {}][EPOCH {}][EVAL] loss: {:.4f}, score: {:.4f}'.format(fold, epoch, loss, score))
        writer.add_scalar('loss', loss, global_step=epoch)
        writer.add_scalar('score', score, global_step=epoch)
        writer.add_image('image', torchvision.utils.make_grid(images[:32], normalize=True), global_step=epoch)
        writer.add_image('thresholds', plot.transpose((2, 0, 1)), global_step=epoch)

        return score


def train_fold(fold, minima):
    train_indices, eval_indices = indices_for_fold(fold, len(train_data))

    train_dataset = TrainEvalDataset(train_data.iloc[train_indices], transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True,
        num_workers=args.workers)  # TODO: all args

    eval_dataset = TrainEvalDataset(train_data.iloc[eval_indices], transform=eval_transform)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, num_workers=args.workers)  # TODO: all args

    model = Model(NUM_CLASSES)
    model = model.to(DEVICE)
    optimizer = build_optimizer(args.opt, model.parameters(), 0., args.beta[-1], weight_decay=args.weight_decay)
    scheduler = OneCycleScheduler(
        optimizer,
        lr=(minima['lr'] / 25, minima['lr']),
        beta=args.beta,
        max_steps=len(train_data_loader) * args.epochs,
        annealing=args.anneal)

    best_score = 0
    for epoch in range(args.epochs):
        train_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data_loader=train_data_loader,
            fold=fold,
            epoch=epoch)
        score = eval_epoch(
            model=model,
            data_loader=eval_data_loader,
            fold=fold,
            epoch=epoch)

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), './model_{}.pth'.format(fold))


def build_submission(folds, threshold):
    with torch.no_grad():
        predictions = 0.

        for fold in folds:
            fold_predictions, fold_ids = predict_on_test_using_fold(fold)
            predictions = predictions + fold_predictions.sigmoid()
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
    test_dataset = TestDataset(transform=eval_transform)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.workers)  # TODO: all args

    model = Model(NUM_CLASSES)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load('./model_{}.pth'.format(fold)))

    model.eval()
    with torch.no_grad():
        fold_predictions = []
        fold_ids = []
        for images, ids in tqdm(test_data_loader, desc='fold {} inference'.format(fold)):
            images = images.to(DEVICE)
            logits = model(images)
            fold_predictions.append(logits)
            fold_ids.extend(ids)

            if args.debug:
                break

        fold_predictions = torch.cat(fold_predictions, 0)

    return fold_predictions, fold_ids


def predict_on_eval_using_fold(fold):
    _, eval_indices = indices_for_fold(fold, len(train_data))

    eval_dataset = TrainEvalDataset(train_data.iloc[eval_indices], transform=eval_transform)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, num_workers=args.workers)  # TODO: all args

    model = Model(NUM_CLASSES)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load('./model_{}.pth'.format(fold)))

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

            if args.debug:
                break

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
        threshold, score, plot = find_threshold_global(input=predictions, target=targets)

        print('threshold: {:.4f}, score: {:.4f}'.format(threshold, score))

        return threshold


# TODO: check FOLDS usage

def main():
    id_to_class = list(pd.read_csv(os.path.join(args.dataset_path, 'sample_submission.csv')).columns[1:])
    class_to_id = {c: i for i, c in enumerate(id_to_class)}

    train_data = pd.read_csv(os.path.join(args.dataset_path, 'train_curated.csv'))
    train_data['fname'] = train_data['fname'].apply(lambda x: os.path.join(args.dataset_path, 'train_curated', x))
    train_data['labels'] = train_data['labels'].apply(lambda x: [class_to_id[c] for c in x.split(',')])

    minima = find_lr(train_data)
    gc.collect()

    if args.fold is None:
        folds = FOLDS
    else:
        folds = [args.fold]

    for fold in folds:
        train_fold(fold, minima)

    # TODO: check and refine
    threshold = find_threshold_for_folds(folds)
    build_submission(folds, threshold)


if __name__ == '__main__':
    main()
