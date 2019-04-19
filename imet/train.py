import numpy as np
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
from scheduler import OneCycleScheduler
from utils import Mean, seed_everything
from augmentation import SquarePad
from .model import Model

# TODO: check del

FOLDS = list(range(1, 5 + 1))

parser = argparse.ArgumentParser()
parser.add_argument('--experiment-path', type=str, default='./tf_log/imet')
parser.add_argument('--dataset-path', type=str, required=True)
parser.add_argument('--workers', type=int, default=os.cpu_count())
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--fold', type=int, choices=FOLDS)
parser.add_argument('--image-size', type=int, default=128)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--annealing', type=str, choices=['linear', 'cosine'], default='linear')
parser.add_argument('--aug', type=str, choices=['low', 'med', 'med+color', 'hard', 'pad'], default='med')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

seed_everything(args.seed)

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
        path = './imet/dog.jpg'

    image = Image.open(path)

    return image


class EWA(object):
    def __init__(self, beta=0.9):
        self.beta = beta
        self.step = 0
        self.average = 0

    def update(self, value):
        self.step += 1
        self.average = self.beta * self.average + (1 - self.beta) * value

    def compute(self):
        return self.average / (1 - self.beta**self.step)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        target = target.float()
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size()) == 2:
            loss = loss.sum(dim=1)
        return loss.mean()


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

    # plt.plot(thresholds, scores)
    # plt.axvline(threshold)
    # plt.title('score: {:.4f}, threshold: {:.4f}'.format(score.item(), threshold))
    # plt.show()

    return threshold, score


def find_threshold_class(input, target, initial):
    threshold = torch.full((NUM_CLASSES,), initial).to(input.device)
    steps = []
    for _ in tqdm(range(50), desc='threshold search'):
        for i in np.random.permutation(NUM_CLASSES):
            r = torch.tensor([threshold[i] - 0.01, threshold[i], threshold[i] + 0.01])
            space = threshold.view(1, NUM_CLASSES).repeat(r.size(0), 1)
            space[:, i] = r
            scores = compute_score(input=input.unsqueeze(0), target=target.unsqueeze(0), threshold=space.unsqueeze(1))
            scores = scores.mean(-1)
            threshold[i] = r[scores.argmax()]

        score = compute_score(input=input, target=target, threshold=threshold).mean()
        steps.append(score)

    # plt.plot(steps)
    # plt.show()

    # plt.hist(threshold.cpu(), bins=50)
    # plt.title('score: {:.4f}, threshold mean: {:.4f}, threshold std: {:.4f}'.format(
    #     score.item(), threshold.mean(), threshold.std()))
    # plt.show()

    return threshold


NUM_CLASSES = len(classes)
ARCH = 'seresnext50'
OPT = 'adam'
LOSS_SMOOTHING = 0.9
LOSS = [
    # f2_loss,
    # F.binary_cross_entropy_with_logits,
    FocalLoss(),
    # hinge_loss,
]
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


to_tensor_and_norm = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if args.aug == 'low':
    train_transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.RandomHorizontalFlip(),
        to_tensor_and_norm,
    ])
    eval_transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        to_tensor_and_norm,
    ])
elif args.aug == 'med':
    train_transform = T.Compose([
        T.RandomResizedCrop((args.image_size, args.image_size), scale=(1., 1.), ratio=(3. / 4., 4. / 3.)),
        T.RandomHorizontalFlip(),
        to_tensor_and_norm,
    ])
    eval_transform = T.Compose([
        T.Resize(args.image_size),
        T.CenterCrop(args.image_size),
        to_tensor_and_norm,
    ])
elif args.aug == 'med+color':
    train_transform = T.Compose([
        T.RandomResizedCrop((args.image_size, args.image_size), scale=(1., 1.), ratio=(3. / 4., 4. / 3.)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        to_tensor_and_norm,
    ])
    eval_transform = T.Compose([
        T.Resize(args.image_size),
        T.CenterCrop(args.image_size),
        to_tensor_and_norm,
    ])
elif args.aug == 'hard':
    scale = (0.6, 1.0)
    # image_size_corrected = round(args.image_size * (1 / np.mean(scale).item()))
    image_size_corrected = args.image_size

    train_transform = T.Compose([
        T.RandomResizedCrop((args.image_size, args.image_size), scale=scale, ratio=(3. / 4., 4. / 3.)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        to_tensor_and_norm,
    ])
    # TODO: correct eval size?
    eval_transform = T.Compose([
        T.Resize(image_size_corrected),
        T.CenterCrop(image_size_corrected),
        to_tensor_and_norm,
    ])
elif args.aug == 'pad':
    image_size_corrected = round(args.image_size * (1 / 0.8))

    train_transform = T.Compose([
        SquarePad(padding_mode='edge'),
        T.Resize(image_size_corrected),
        T.RandomCrop(args.image_size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        to_tensor_and_norm,
    ])
    eval_transform = T.Compose([
        SquarePad(padding_mode='edge'),
        T.Resize(image_size_corrected),
        T.CenterCrop(args.image_size),
        to_tensor_and_norm,
    ])
else:
    raise AssertionError('invalid AUG {}'.format(args.aug))


def build_optimizer(parameters, lr, weight_decay):
    if OPT == 'adam':
        return torch.optim.Adam(parameters, lr, weight_decay=weight_decay)
    elif OPT == 'sgd':
        return torch.optim.SGD(parameters, lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise AssertionError('invalid OPT {}'.format(OPT))


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def find_lr():
    train_dataset = TrainEvalDataset(train_data, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True,
        num_workers=args.workers)  # TODO: all args

    min_lr = 1e-8
    max_lr = 10.
    gamma = (max_lr / min_lr)**(1 / len(train_data_loader))

    stp = 0
    cur_lr = min_lr
    lrs = []
    lss = []
    sls = []
    ewa = EWA(beta=LOSS_SMOOTHING)

    minima = {
        'loss': np.inf,
        'lr': min_lr
    }

    model = Model(ARCH, NUM_CLASSES)
    model = model.to(DEVICE)
    optimizer = build_optimizer(model.parameters(), min_lr, weight_decay=args.weight_decay)

    writer = SummaryWriter(os.path.join(args.experiment_path, 'lr_search'))

    model.train()
    for images, labels, ids in tqdm(train_data_loader, desc='lr search'):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)

        loss = compute_loss(input=logits, target=labels)

        ewa.update(loss.data.cpu().numpy().mean())
        lrs.append(cur_lr)
        lss.append(loss.data.cpu().numpy().mean())
        sls.append(ewa.compute())
        if ewa.compute() < minima['loss']:
            minima['loss'] = ewa.compute()
            minima['lr'] = cur_lr
        if minima['loss'] * 4 < ewa.compute():
            break
        stp += 1

        cur_lr = min_lr * gamma**stp
        set_lr(optimizer, cur_lr)

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        writer.add_scalar('loss', loss.mean().data.cpu().numpy(), global_step=stp)

        if args.debug:
            break

    return minima


def indices_for_fold(fold, dataset_size):
    kfold = KFold(len(FOLDS), shuffle=True, random_state=args.seed)
    splits = list(kfold.split(np.zeros(dataset_size)))
    train_indices, eval_indices = splits[fold - 1]
    assert len(train_indices) + len(eval_indices) == dataset_size

    return train_indices, eval_indices


def train(model, optimizer, scheduler, data_loader, fold, epoch):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'train', 'fold_{}'.format(fold)))

    metrics = {
        'loss': Mean(),
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
        writer.add_scalar('lr', lr, global_step=epoch)
        writer.add_scalar('beta', beta, global_step=epoch)
        writer.add_image('image', torchvision.utils.make_grid(images[:32], normalize=True), global_step=epoch)


def eval(model, data_loader, fold, epoch):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'eval', 'fold{}'.format(fold)))

    metrics = {
        'loss': Mean(),
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
        threshold, score = find_threshold_global(input=predictions, target=targets)

        print('[FOLD {}][EPOCH {}][EVAL] loss: {:.4f}, score: {:.4f}'.format(fold, epoch, loss, score))
        writer.add_scalar('loss', loss, global_step=epoch)
        writer.add_scalar('score', score, global_step=epoch)
        writer.add_image('image', torchvision.utils.make_grid(images[:32], normalize=True), global_step=epoch)

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

    model = Model(ARCH, NUM_CLASSES)
    model = model.to(DEVICE)
    optimizer = build_optimizer(model.parameters(), 0., weight_decay=args.weight_decay)
    scheduler = OneCycleScheduler(
        optimizer, lr=(minima['lr'] / 10 / 25, minima['lr'] / 10), beta=(0.95, 0.85),
        max_steps=len(train_data_loader) * args.epochs, annealing=args.annealing)

    best_score = 0
    for epoch in range(args.epochs):
        train(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data_loader=train_data_loader,
            fold=fold,
            epoch=epoch)
        score = eval(
            model=model,
            data_loader=eval_data_loader,
            fold=fold,
            epoch=epoch)

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), './model_{}.pth'.format(fold))


def build_submission(threshold):
    with torch.no_grad():
        predictions = 0.

        for fold in FOLDS:
            fold_predictions, fold_ids = predict_on_test_using_fold(fold)
            predictions = predictions + fold_predictions.sigmoid()
            ids = fold_ids

        predictions = predictions / len(FOLDS)
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

    model = Model(ARCH, NUM_CLASSES)
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

    model = Model(ARCH, NUM_CLASSES)
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


def find_threshold_for_folds():
    with torch.no_grad():
        targets = []
        predictions = []
        for fold in FOLDS:
            fold_targets, fold_predictions, fold_ids = predict_on_eval_using_fold(fold)
            targets.append(fold_targets)
            predictions.append(fold_predictions)

        predictions = torch.cat(predictions, 0)
        targets = torch.cat(targets, 0)
        threshold, score = find_threshold_global(input=predictions, target=targets)

        print('threshold: {:.4f}, score: {:.4f}'.format(threshold, score))

        return threshold


def main():
    minima = find_lr()

    if args.fold is None:
        for fold in FOLDS:
            train_fold(fold, minima)
    else:
        train_fold(args.fold, minima)

    # threshold = find_threshold_for_folds()
    # build_submission(threshold)


if __name__ == '__main__':
    main()
