import numpy as np
import gc
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import argparse
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold
from lr_scheduler import OneCycleScheduler
from optim import AdamW
import utils
from transform import SquarePad
from .model import Model
from loss import lsep_loss
import torch.nn.functional as F

# TODO: sgd
# TODO: check del
# TODO: try largest lr before diverging
# TODO: check all plots rendered
# TODO: adamw
# TODO: better minimum for lr


# TODO: save all data in folder
# TODO: full debug run

FOLDS = list(range(1, 5 + 1))

parser = argparse.ArgumentParser()
parser.add_argument('--experiment-path', type=str, default='./tf_log/imet')
parser.add_argument('--dataset-path', type=str, required=True)
parser.add_argument('--workers', type=int, default=os.cpu_count())
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--fold', type=int, choices=FOLDS)
parser.add_argument('--image-size', type=int, default=224)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--focal-gamma', type=float, default=2.)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--beta', type=float, nargs=2, default=(0.95, 0.85))
parser.add_argument('--anneal', type=str, choices=['linear', 'cosine'], default='linear')
parser.add_argument('--aug', type=str, choices=['resize', 'crop', 'pad'], default='pad')
parser.add_argument('--aug-aspect', action='store_true')
parser.add_argument('--crop-scale', type=float, default=224 / 256)
parser.add_argument('--opt', type=str, choices=['adam', 'adamw', 'momentum'], default='adam')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

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
        for l in row['attribute_ids']:  # TODO:
            label[l] = 1.

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


def compute_loss(input, target):
    logits, thresholds = input.split(input.shape[1] // 2, 1)

    class_loss = lsep_loss(input=logits, target=target)
    thresh_loss = F.binary_cross_entropy_with_logits(input=logits - thresholds, target=target, reduction='sum')

    # TODO: normalize by batch?
    return (class_loss + thresh_loss / input.size(0)) / 2


def output_to_logits(input):
    logits, thresholds = input.split(input.shape[1] // 2, 1)

    return logits - thresholds


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


NUM_CLASSES = len(classes)
ARCH = 'seresnext50'
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


image_size_corrected = round(args.image_size * (1 / args.crop_scale))

if args.aug == 'resize':
    resize = T.Resize((image_size_corrected, image_size_corrected))

    train_transform = resize
    eval_transform = resize
elif args.aug == 'crop':
    resize = T.Resize(image_size_corrected)

    train_transform = resize
    eval_transform = resize
elif args.aug == 'pad':
    pad_and_resize = T.Compose([
        SquarePad(padding_mode='edge'),
        T.Resize(image_size_corrected),
    ])

    train_transform = pad_and_resize
    eval_transform = pad_and_resize
else:
    raise AssertionError('invalid aug {}'.format(args.aug))

if args.aug_aspect:
    random_crop = T.RandomResizedCrop(
        args.image_size, scale=(args.crop_scale, args.crop_scale), ratio=(3. / 4., 4. / 3.))
else:
    random_crop = T.RandomCrop(args.image_size)

to_tensor_and_norm = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_transform = T.Compose([
    train_transform,
    random_crop,
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    to_tensor_and_norm,
])
eval_transform = T.Compose([
    eval_transform,
    T.CenterCrop(args.image_size),
    to_tensor_and_norm,
])
test_transform = eval_transform


# test_transform = T.Compose([
#     SquarePad(padding_mode='edge'),
#     T.Resize(image_size_corrected),
#     T.TenCrop(args.image_size),
#     T.Lambda(lambda xs: torch.stack([to_tensor_and_norm(x) for x in xs], 0)),
# ])
# test_transform = T.Compose([
#     SquarePad(padding_mode='edge'),
#     T.Resize(image_size_corrected),
#     T.TenCrop(args.image_size),
#     T.Lambda(lambda xs: torch.stack([to_tensor_and_norm(x) for x in xs], 0)),
# ])


# TODO: should use top momentum to pick best lr?
def build_optimizer(optimizer, parameters, lr, beta, weight_decay):
    if optimizer == 'adam':
        return torch.optim.Adam(parameters, lr, betas=(beta, 0.999), weight_decay=weight_decay)
    elif optimizer == 'adamw':
        return AdamW(parameters, lr, betas=(beta, 0.999), weight_decay=weight_decay)
    elif optimizer == 'momentum':
        return torch.optim.SGD(parameters, lr, momentum=beta, weight_decay=weight_decay)
    else:
        raise AssertionError('invalid OPT {}'.format(optimizer))


def indices_for_fold(fold, dataset_size):
    kfold = KFold(len(FOLDS), shuffle=True, random_state=args.seed)
    splits = list(kfold.split(np.zeros(dataset_size)))
    train_indices, eval_indices = splits[fold - 1]
    assert len(train_indices) + len(eval_indices) == dataset_size

    return train_indices, eval_indices


def mixup(images, labels):
    bs = images.size(0)
    images_a, images_b = torch.split(images, bs // 2)
    labels_a, labels_b = torch.split(labels, bs // 2)

    r = torch.rand(())
    images = r * images_a + (1 - r) * images_b
    labels = r * labels_a + (1 - r) * labels_b

    return images, labels


def find_lr():
    train_dataset = TrainEvalDataset(train_data, transform=train_transform)
    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=args.workers)

    min_lr = 1e-8
    max_lr = 10.
    gamma = (max_lr / min_lr)**(1 / len(data_loader))

    lrs = []
    losses = []
    lim = None

    model = Model(ARCH, NUM_CLASSES * 2)
    model = model.to(DEVICE)
    optimizer = build_optimizer(args.opt, model.parameters(), min_lr, args.beta[-1], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    model.train()
    for images, labels, ids in tqdm(data_loader, desc='lr search'):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        # images, labels = mixup(images, labels)
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
        # images, labels = mixup(images, labels)
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

    model.eval()
    with torch.no_grad():
        predictions = []
        targets = []

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
        train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=args.workers)

    eval_dataset = TrainEvalDataset(train_data.iloc[eval_indices], transform=eval_transform)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, num_workers=args.workers)

    model = Model(ARCH, NUM_CLASSES * 2)
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
            torch.save(model.state_dict(), os.path.join(args.experiment_path, 'model_{}.pth'.format(fold)))


def build_submission(folds, threshold):
    with torch.no_grad():
        predictions = 0.

        for fold in folds:
            fold_predictions, fold_ids = predict_on_test_using_fold(fold)
            fold_predictions = output_to_logits(fold_predictions)
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
        submission.to_csv(os.path.join(args.experiment_path, 'submission.csv'), index=False)


def predict_on_test_using_fold(fold):
    test_dataset = TestDataset(transform=test_transform)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.workers)

    model = Model(ARCH, NUM_CLASSES * 2)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(args.experiment_path, 'model_{}.pth'.format(fold))))

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
        eval_dataset, batch_size=args.batch_size, num_workers=args.workers)

    model = Model(ARCH, NUM_CLASSES * 2)
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
        threshold, score, plot = find_threshold_global(input=predictions, target=targets)

        print('threshold: {:.4f}, score: {:.4f}'.format(threshold, score))

        return threshold


# TODO: check FOLDS usage

def main():
    # TODO: refactor seed
    utils.seed_everything(args.seed)

    minima = find_lr()
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
