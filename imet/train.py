import numpy as np
import shutil
import gc
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
import torch
import torch.utils
import torch.utils.data
import torch.distributions
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import argparse
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from lr_scheduler import OneCycleScheduler, CyclicLR
import lr_scheduler_wrapper
from optim import AdamW
import utils
from transform import SquarePad, RatioPad, Cutout
from .model import Model
from loss import FocalLoss, bce_loss, lsep_loss, f2_loss, lovasz_loss
from config import Config

fail  # TODO: modify seeds

# TODO: try largest lr before diverging
# TODO: check all plots rendered
# TODO: better minimum for lr

FOLDS = list(range(1, 5 + 1))
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser()
parser.add_argument('--config-path', type=str, required=True)
parser.add_argument('--experiment-path', type=str, default='./tf_log/imet')
parser.add_argument('--dataset-path', type=str, required=True)
parser.add_argument('--workers', type=int, default=os.cpu_count())
parser.add_argument('--fold', type=int, choices=FOLDS)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()
config = Config.from_yaml(args.config_path)
shutil.copy(args.config_path, utils.mkdir(args.experiment_path))

train_data = pd.read_csv(os.path.join(args.dataset_path, 'train.csv'))
train_data['attribute_ids'] = train_data['attribute_ids'].apply(lambda s: [int(x) for x in s.split()])

classes = pd.read_csv(os.path.join(args.dataset_path, 'labels.csv'))
id_to_class = {row['attribute_id']: row['attribute_name'] for _, row in classes.iterrows()}
class_to_id = {id_to_class[k]: k for k in id_to_class}


class TrainEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data.iloc[i]

        image = load_image(os.path.join(args.dataset_path, 'train/{}.png'.format(row['id'])))
        if self.transform is not None:
            image = self.transform(image)

        label = np.zeros(NUM_CLASSES, dtype=np.float32)
        label[row['attribute_ids']] = 1.

        return image, label, row['id']


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.data = os.listdir(os.path.join(args.dataset_path, 'test'))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path = self.data[i]
        id = os.path.splitext(path)[0]

        image = load_image(os.path.join(args.dataset_path, 'test', path))
        if self.transform is not None:
            image = self.transform(image)

        return image, id


def load_image(path):
    if args.debug:
        path = './imet/sample.jpg'

    image = Image.open(path)

    return image


def draw_errors(images, true, pred):
    def draw_text(draw, x, ylim, lines, fill):
        font = ImageFont.truetype('./imet/Droid+Sans+Mono+Awesome.ttf', size=10)

        y = ylim[0]
        for line in lines:
            size = draw.textsize(line, font=font)
            if y + size[1] >= ylim[1]:
                break

            draw.rectangle(((x, y), (x + size[0], y + size[1])), fill=(0, 0, 0))
            draw.text((x, y), line, font=font, fill=fill)

            y += size[1] + 2

    def draw_error(image, t, p):
        image = image * std + mean
        image = image.data.cpu().numpy()
        image = image.transpose((1, 2, 0))
        image = (image * 255).round().astype(np.uint8)

        fp = (1 - t) * p
        fn = t * (1 - p)
        t = [id_to_class[id.item()] for id in t.nonzero().data.cpu().numpy()]
        fp = [id_to_class[id.item()] for id in fp.nonzero().data.cpu().numpy()]
        fn = [id_to_class[id.item()] for id in fn.nonzero().data.cpu().numpy()]

        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        draw_text(draw, 5, (5, image.size[1] // 2 - 5), fp, fill=(0, 255, 0))
        draw_text(draw, 5, (image.size[1] // 2 + 5, image.size[1] - 5), fn, fill=(255, 0, 0))
        draw_text(draw, image.size[0] // 2 + 5, (image.size[1] // 2 + 5, image.size[1] - 5), t, fill=(0, 0, 255))

        image = to_tensor_and_norm(image)
        image = image.to(DEVICE)

        return image

    mean = torch.tensor(MEAN).view(3, 1, 1).to(DEVICE)
    std = torch.tensor(STD).view(3, 1, 1).to(DEVICE)

    images = [draw_error(image, t, p) for image, t, p in zip(images, true, pred)]
    images = torch.stack(images, 0)

    return images


def lsepfoc_loss(input, target):
    return lsep_loss(input=input, target=target) + FocalLoss(gamma=config.loss.focal.gamma)(input=input, target=target)


def compute_loss(input, target, smoothing):
    if config.loss.type == 'focal':
        compute_class_loss = FocalLoss(gamma=config.loss.focal.gamma)
    elif config.loss.type == 'f2':
        compute_class_loss = f2_loss
    elif config.loss.type == 'lsep':
        compute_class_loss = lsep_loss
    elif config.loss.type == 'lsepfoc':
        compute_class_loss = lsepfoc_loss
    elif config.loss.type == 'lovasz':
        compute_class_loss = lovasz_loss
    else:
        raise AssertionError('invalid loss {}'.format(config.loss.type))

    if smoothing is not None:
        target = (1 - smoothing) * target + smoothing / 2

    if config.model.predict_thresh:
        logits, thresholds = input.split(input.size(-1) // 2, -1)

        class_loss = compute_class_loss(input=logits, target=target)
        thresh_loss = bce_loss(input=logits - thresholds, target=target)

        loss = (class_loss + thresh_loss) / 2
    else:
        loss = compute_class_loss(input=input, target=target)

    assert loss.dim() == 0

    return loss


def output_to_logits(input):
    if config.model.predict_thresh:
        logits, thresholds = input.split(input.size(-1) // 2, -1)
        logits = logits - thresholds
    else:
        logits = input

    return logits


def compute_score(input, target, threshold=0.5):
    input = output_to_logits(input)
    input = (input.sigmoid() > threshold).float()

    tp = (target * input).sum(-1)
    # tn = ((1 - target) * (1 - input)).sum(-1)
    fp = ((1 - target) * input).sum(-1)
    fn = (target * (1 - input)).sum(-1)

    p = tp / (tp + fp)
    r = tp / (tp + fn)

    beta_sq = 2**2
    f2 = (1 + beta_sq) * p * r / (beta_sq * p + r)
    f2[f2 != f2] = 0.

    return f2


def find_threshold_global(input, target):
    thresholds = np.arange(0.01, 1 - 0.01, 0.01)
    scores = [compute_score(input=input, target=target, threshold=t).mean()
              for t in tqdm(thresholds, desc='threshold search')]
    threshold = thresholds[np.argmax(scores)]
    score = scores[np.argmax(scores)]

    fig = plt.figure()
    plt.plot(thresholds, scores)
    plt.axvline(threshold)
    plt.title('score: {:.4f}, threshold: {:.4f}'.format(score.item(), threshold))

    return threshold, score, fig


# def find_threshold_global(input, target):
#     freq = target.mean(0)
#     indices = freq.argsort()
#     threshold = torch.zeros(indices.size()).to(DEVICE)
#
#     # global
#     i_global = indices[:-100]
#     ts = np.arange(0.01, 1 - 0.01, 0.01)
#     scores = []
#     for t in ts:
#         score = compute_score(input=input[:, i_global], target=target[:, i_global], threshold=t).mean()
#         scores.append(score)
#     threshold[i_global] = ts[np.argmax(scores)]
#
#     # class
#     i_class = indices[-100:]
#     for i in tqdm(i_class, desc='threshold search'):
#         i = [i]
#         ts = np.arange(0.01, 1 - 0.01, 0.01)
#         scores = []
#         for t in ts:
#             score = compute_score(input=input[:, i], target=target[:, i], threshold=t).mean()
#             scores.append(score)
#         threshold[i] = ts[np.argmax(scores)]
#
#     # eval
#     score = compute_score(input=input, target=target, threshold=threshold).mean()
#     fig = plt.figure()
#
#     return threshold, score, fig


NUM_CLASSES = len(classes)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
# TODO: cutout
# TODO: larger model
# TODO: load image as jpeg
# TODO: min 1 tag?
# TODO: pick threshold to match ratio
# TODO: compute smoothing beta from batch size and num steps
# TODO: speedup image loading
# TODO: smart sampling
# TODO: better threshold search (step, epochs)
# TODO: weight standartization
# TODO: build sched for lr find


image_size_corrected = round(config.image_size * (1 / config.aug.crop_scale))

if config.aug.type == 'resize':
    resize = T.Resize((image_size_corrected, image_size_corrected))

    train_transform = resize
    eval_transform = resize
    test_transform = resize
elif config.aug.type == 'crop':
    resize = T.Resize(image_size_corrected)

    train_transform = resize
    eval_transform = resize
    test_transform = resize
elif config.aug.type == 'pad':
    pad_and_resize = T.Compose([
        SquarePad(padding_mode='edge'),
        T.Resize(image_size_corrected),
    ])

    train_transform = pad_and_resize
    eval_transform = pad_and_resize
    test_transform = pad_and_resize
elif config.aug.type == 'rpad':
    pad_and_resize = T.Compose([
        RatioPad(padding_mode='edge'),
        T.Resize(image_size_corrected),
    ])

    train_transform = pad_and_resize
    eval_transform = pad_and_resize
    test_transform = pad_and_resize
else:
    raise AssertionError('invalid aug {}'.format(config.aug.type))

if config.aug.scale:
    crop_scale = (config.aug.crop_scale**2 * 2 - 1, 1.)
    assert config.aug.crop_scale == np.sqrt(np.mean(crop_scale))
    random_crop = T.RandomResizedCrop(config.image_size, scale=crop_scale)
else:
    random_crop = T.RandomCrop(config.image_size)

to_tensor_and_norm = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])
train_transform = T.Compose([
    train_transform,
    random_crop,
    T.RandomHorizontalFlip(),
    T.ColorJitter(
        brightness=config.aug.color.brightness,
        contrast=config.aug.color.contrast,
        saturation=config.aug.color.saturation,
        hue=config.aug.color.hue),
    T.RandomGrayscale(p=config.aug.grayscale),
    to_tensor_and_norm,
    Cutout(n_holes=config.aug.cutout.n_holes, length=round(config.image_size * config.aug.cutout.length)),
])
eval_transform = T.Compose([
    eval_transform,
    T.CenterCrop(config.image_size),
    to_tensor_and_norm,
])
test_transform = T.Compose([
    test_transform,
    T.TenCrop(config.image_size),
    T.Lambda(lambda xs: torch.stack([to_tensor_and_norm(x) for x in xs], 0))
])


def build_optimizer(optimizer, parameters, lr, beta, weight_decay):
    if optimizer == 'adam':
        return torch.optim.Adam(parameters, lr, betas=(beta, 0.999), weight_decay=weight_decay)
    elif optimizer == 'adamw':
        return AdamW(parameters, lr, betas=(beta, 0.999), weight_decay=weight_decay)
    elif optimizer == 'momentum':
        return torch.optim.SGD(parameters, lr, momentum=beta, weight_decay=weight_decay, nesterov=True)
    else:
        raise AssertionError('invalid OPT {}'.format(optimizer))


def indices_for_fold(fold, dataset_size):
    kfold = KFold(len(FOLDS), shuffle=True, random_state=config.seed)
    splits = list(kfold.split(np.zeros(dataset_size)))
    train_indices, eval_indices = splits[fold - 1]
    assert len(train_indices) + len(eval_indices) == dataset_size

    return train_indices, eval_indices


# def indices_for_fold(fold, dataset_size):
#     labels = np.zeros((dataset_size, NUM_CLASSES))
#     for i in tqdm(range(dataset_size), desc='stratification'):
#         row = train_data.iloc[i]
#         labels[i, row['attribute_ids']] = 1.
#
#     kfold = MultilabelStratifiedKFold(len(FOLDS), shuffle=True, random_state=config.seed)
#     splits = list(kfold.split(np.zeros(dataset_size), labels))
#     train_indices, eval_indices = splits[fold - 1]
#     assert len(train_indices) + len(eval_indices) == dataset_size
#
#     return train_indices, eval_indices


class MixupDataLoader(object):
    def __init__(self, data_loader, alpha):
        self.data_loader = data_loader
        self.dist = torch.distributions.beta.Beta(alpha, alpha)

        # TODO: handle ids
        # TODO: sample beta for each sample
        # TODO: speedup

    def __iter__(self):
        for (images_1, labels_1, ids_1), (images_2, labels_2, ids_2) in zip(self.data_loader, self.data_loader):
            lam = self.dist.sample().to(DEVICE)

            images = lam * images_1.to(DEVICE) + (1 - lam) * images_2.to(DEVICE)
            labels = lam * labels_1.to(DEVICE) + (1 - lam) * labels_2.to(DEVICE)
            ids = ids_1 if lam > 0.5 else ids_2

            yield images, labels, ids

    def __len__(self):
        return len(self.data_loader)


def find_lr():
    train_dataset = TrainEvalDataset(train_data, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, drop_last=True, shuffle=True, num_workers=args.workers)
    if config.mixup is not None:
        train_data_loader = MixupDataLoader(train_data_loader, config.mixup)

    min_lr = 1e-7
    max_lr = 10.
    gamma = (max_lr / min_lr)**(1 / len(train_data_loader))

    lrs = []
    losses = []
    lim = None

    model = Model(config.model, NUM_CLASSES)
    model = model.to(DEVICE)
    optimizer = build_optimizer(
        config.opt.type, model.parameters(), min_lr, config.opt.beta, weight_decay=config.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    model.train()
    for images, labels, ids in tqdm(train_data_loader, desc='lr search'):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)

        loss = compute_loss(input=logits, target=labels, smoothing=config.label_smooth)

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

        if args.debug:
            break

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

        loss = compute_loss(input=logits, target=labels, smoothing=config.label_smooth)
        metrics['loss'].update(loss.data.cpu().numpy())

        lr, beta = scheduler.get_lr()
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        scheduler.step()

        if args.debug:
            break

    with torch.no_grad():
        loss = metrics['loss'].compute_and_reset()

        print('[FOLD {}][EPOCH {}][TRAIN] loss: {:.4f}'.format(fold, epoch, loss))
        writer.add_scalar('loss', loss, global_step=epoch)
        writer.add_scalar('learning_rate', lr, global_step=epoch)
        writer.add_scalar('beta', beta, global_step=epoch)
        writer.add_image('image', torchvision.utils.make_grid(images[:32], normalize=True), global_step=epoch)


def eval_epoch(model, data_loader, fold, epoch):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'fold{}'.format(fold), 'eval'))

    metrics = {
        'loss': utils.Mean(),
    }

    model.eval()
    with torch.no_grad():
        predictions = []
        targets = []

        for images, labels, ids in tqdm(data_loader, desc='epoch {} evaluation'.format(epoch)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)

            targets.append(labels)
            predictions.append(logits)

            loss = compute_loss(input=logits, target=labels, smoothing=config.label_smooth)
            metrics['loss'].update(loss.data.cpu().numpy())

            if args.debug:
                break

        loss = metrics['loss'].compute_and_reset()

        predictions = torch.cat(predictions, 0)
        targets = torch.cat(targets, 0)
        threshold, score, fig = find_threshold_global(input=predictions, target=targets)

        scores = compute_score(input=predictions, target=targets, threshold=threshold)
        indices = scores.argsort()[:32]
        failure = [data_loader.dataset[i.item()][0] for i in indices]
        failure = torch.stack(failure, 0).to(DEVICE)
        failure = draw_errors(
            failure,
            true=(output_to_logits(predictions[indices]).sigmoid() > threshold).float(),
            pred=targets[indices])

        print('[FOLD {}][EPOCH {}][EVAL] loss: {:.4f}, score: {:.4f}'.format(fold, epoch, loss, score))
        writer.add_scalar('loss', loss, global_step=epoch)
        writer.add_scalar('score', score, global_step=epoch)
        writer.add_image('image', torchvision.utils.make_grid(images[:32], normalize=True), global_step=epoch)
        writer.add_image('failure', torchvision.utils.make_grid(failure, normalize=True), global_step=epoch)
        writer.add_figure('thresholds', fig, global_step=epoch)

        return score


def train_fold(fold, lr):
    train_indices, eval_indices = indices_for_fold(fold, len(train_data))

    train_dataset = TrainEvalDataset(train_data.iloc[train_indices], transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, drop_last=True, shuffle=True, num_workers=args.workers)
    if config.mixup is not None:
        train_data_loader = MixupDataLoader(train_data_loader, config.mixup)

    eval_dataset = TrainEvalDataset(train_data.iloc[eval_indices], transform=eval_transform)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=config.batch_size, num_workers=args.workers)

    model = Model(config.model, NUM_CLASSES)
    model = model.to(DEVICE)
    optimizer = build_optimizer(
        config.opt.type, model.parameters(), lr, config.opt.beta, weight_decay=config.opt.weight_decay)

    if config.sched.type == 'onecycle':
        scheduler = lr_scheduler_wrapper.StepWrapper(
            OneCycleScheduler(
                optimizer,
                lr=(lr / 20, lr),
                beta=config.sched.onecycle.beta,
                max_steps=len(train_data_loader) * config.epochs,
                annealing=config.sched.onecycle.anneal))
    elif config.sched.type == 'cyclic':
        scheduler = lr_scheduler_wrapper.StepWrapper(
            CyclicLR(
                optimizer,
                0.,
                lr,
                step_size_up=len(train_data_loader),
                step_size_down=len(train_data_loader),
                mode='triangular2',
                cycle_momentum=True,
                base_momentum=config.sched.cyclic.beta[1],
                max_momentum=config.sched.cyclic.beta[0]))
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
        score = eval_epoch(
            model=model,
            data_loader=eval_data_loader,
            fold=fold,
            epoch=epoch)
        gc.collect()

        scheduler.step_epoch()
        scheduler.step_score(score)

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(args.experiment_path, 'model_{}.pth'.format(fold)))


def build_submission(folds, threshold):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'test'))

    with torch.no_grad():
        predictions = 0.

        for fold in folds:
            fold_predictions, fold_ids = predict_on_test_using_fold(fold)
            fold_predictions = output_to_logits(fold_predictions)
            fold_predictions = fold_predictions.sigmoid().mean(1)

            writer.add_histogram('distribution', fold_predictions, global_step=fold)

            predictions = predictions + fold_predictions
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
        submission.to_csv(os.path.join(args.experiment_path, 'submission.csv'), index=False)


def predict_on_test_using_fold(fold):
    test_dataset = TestDataset(transform=test_transform)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size // 2, num_workers=args.workers)

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
            logits = logits.view(b, n, NUM_CLASSES * (1 + config.model.predict_thresh))

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
        eval_dataset, batch_size=config.batch_size, num_workers=args.workers)

    model = Model(config.model, NUM_CLASSES)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(args.experiment_path, 'model_{}.pth'.format(fold))))

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
        threshold, score, _ = find_threshold_global(input=predictions, target=targets)

        print('threshold: {:.4f}, score: {:.4f}'.format(threshold, score))

        return threshold


# TODO: check FOLDS usage

def main():
    # TODO: refactor seed
    utils.seed_python(config.seed)

    if config.opt.lr is None:
        lr = find_lr()
        print('find_lr: {}'.format(lr))
        gc.collect()
    else:
        lr = config.opt.lr

    if args.fold is None:
        folds = FOLDS
    else:
        folds = [args.fold]

    for fold in folds:
        train_fold(fold, lr)

    # TODO: check and refine
    threshold = find_threshold_for_folds(folds)
    build_submission(folds, threshold)


if __name__ == '__main__':
    main()
