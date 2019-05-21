import shutil
import torch.nn.functional as F
import numpy as np
from config import Config
import gc
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import argparse
from tensorboardX import SummaryWriter
from lr_scheduler import OneCycleScheduler
import lr_scheduler_wrapper
from optim import AdamW
import utils
from .dataset import NUM_CLASSES
from frees.metric import calculate_per_class_lwlrap


# TODO: resnext
# TODO: nfft, hop len
# TODO: no mel fiters
# TODO: check how spectras built
# TODO: mixup

# TODO: remove unused code
# TODO: bucketing
# TODO: remove paddding
# TODO: crop signal, not spectra
# TODO: crop to average curated len
# TODO: rename train_eval to train_curated
# TODO: remove CyclicLR
# TODO: cutout
# TODO: resample silence
# TODO: benchmark stft
# TODO: scipy stft
# TODO: try max pool
# TODO: sgd
# TODO: check del
# TODO: try largest lr before diverging
# TODO: check all plots rendered
# TODO: adamw

class Dataset(torch.utils.data.Dataset):
    def __getitem__(self, i):
        rng = np.random.RandomState(i + np.random.randint(len(self)))

        input = rng.standard_normal(NUM_CLASSES) * 2

        indices = rng.permutation(NUM_CLASSES)
        indices = indices[:rng.randint(1, NUM_CLASSES)]
        target = np.zeros(NUM_CLASSES)
        target[indices] = 1

        input = torch.tensor(input).float()
        target = torch.tensor(target).float()

        score = compute_score(input=input[None, :], target=target[None, :])
        score = torch.tensor(score).float()

        features = torch.stack([
            input,
            target,
            input + target,
            input - target,
            input * target,
        ], 0)

        return features, score

    def __len__(self):
        return 100000


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(5, 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 8, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 8, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 8, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 8, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 8, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 8, 2, stride=2),
        )
        # self.pool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Linear(8, 1)

    def forward(self, input):
        # print(input.shape)
        input = self.conv(input)
        # print(input.shape)
        # input = self.pool(input)
        # print(input.shape)
        input = input.squeeze(-1)
        # print(input.shape)
        input = self.output(input)
        # print(input.shape)
        input = input.squeeze(-1)
        # print(input.shape)
        # fail

        return input


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--config-path', type=str, required=True)
parser.add_argument('--experiment-path', type=str, default='./tf_log/frees')
parser.add_argument('--workers', type=int, default=os.cpu_count())
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()
config = Config.from_yaml(args.config_path)
shutil.copy(args.config_path, utils.mkdir(args.experiment_path))


def compute_loss(input, target):
    loss = F.binary_cross_entropy_with_logits(input=input, target=target, reduction='mean')
    assert loss.dim() == 0

    return loss


def compute_score(input, target):
    per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(
        truth=target.data.cpu().numpy(), scores=input.data.cpu().numpy())

    return np.sum(per_class_lwlrap * weight_per_class)


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


# TODO: should use top momentum to pick best lr?
def build_optimizer(optimizer, parameters, lr, beta, weight_decay):
    if optimizer == 'adam':
        return torch.optim.Adam(parameters, lr, betas=(beta, 0.999), weight_decay=weight_decay)
    elif optimizer == 'adamw':
        return AdamW(parameters, lr, betas=(beta, 0.999), weight_decay=weight_decay)
    elif optimizer == 'momentum':
        return torch.optim.SGD(parameters, lr, momentum=beta, weight_decay=weight_decay, nesterov=True)
    elif optimizer == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr, momentum=beta, weight_decay=weight_decay)
    else:
        raise AssertionError('invalid OPT {}'.format(optimizer))


def train_epoch(model, optimizer, scheduler, data_loader, epoch):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'train'))

    metrics = {
        'loss': utils.Mean(),
    }

    model.train()
    for inputs, labels in tqdm(data_loader, desc='epoch {} train'.format(epoch)):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        logits = model(inputs)

        loss = compute_loss(input=logits, target=labels)
        metrics['loss'].update(loss.data.cpu().numpy())

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        scheduler.step()

        if args.debug:
            break

    with torch.no_grad():
        loss = metrics['loss'].compute_and_reset()

        print('[EPOCH {}][TRAIN] loss: {:.4f}'.format(epoch, loss))
        writer.add_scalar('loss', loss, global_step=epoch)
        lr, beta = scheduler.get_lr()
        writer.add_scalar('learning_rate', lr, global_step=epoch)
        writer.add_scalar('beta', beta, global_step=epoch)
        # writer.add_image(
        #     'image',
        #     torchvision.utils.make_grid(images[:32], nrow=get_nrow(images[:32]), normalize=True),
        #     global_step=epoch)
        # writer.add_histogram(
        #     'distribution',
        #     images[:32],
        #     global_step=epoch)
        # writer.add_image(
        #     'weights',
        #     torchvision.utils.make_grid(weights[:32], nrow=get_nrow(weights[:32])),
        #     global_step=epoch)


def eval_epoch(model, data_loader, epoch):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'eval'))

    metrics = {
        'loss': utils.Mean(),
    }

    predictions = []
    targets = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc='epoch {} evaluation'.format(epoch)):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            logits = model(inputs)

            targets.append(labels)
            predictions.append(logits)

            loss = compute_loss(input=logits, target=labels)
            metrics['loss'].update(loss.data.cpu().numpy())

            if args.debug:
                break

        loss = metrics['loss'].compute_and_reset()

        predictions = torch.cat(predictions, 0)
        targets = torch.cat(targets, 0)
        # score = compute_score(input=predictions, target=targets)
        score = loss

        print('[EPOCH {}][EVAL] loss: {:.4f}, score: {:.4f}'.format(epoch, loss, score))
        writer.add_scalar('loss', loss, global_step=epoch)
        writer.add_scalar('score', score, global_step=epoch)
        # writer.add_image(
        #     'image',
        #     torchvision.utils.make_grid(images[:32], nrow=get_nrow(images[:32]), normalize=True),
        #     global_step=epoch)
        # writer.add_histogram(
        #     'distribution',
        #     images[:32],
        #     global_step=epoch)
        # writer.add_image(
        #     'weights',
        #     torchvision.utils.make_grid(weights[:32], nrow=get_nrow(weights[:32])),
        #     global_step=epoch)

        return score


def train(lr):
    train_dataset = Dataset()
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=args.workers)  # TODO: all args

    eval_dataset = Dataset()
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.batch_size // 2,
        num_workers=args.workers)  # TODO: all args

    model = Model()
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
            torch.optim.lr_scheduler.CyclicLR(
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
                optimizer, mode='min', factor=0.5, patience=3, verbose=True))
    else:
        raise AssertionError('invalid sched {}'.format(config.sched.type))

    best_score = 0
    for epoch in range(config.epochs):

        train_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data_loader=train_data_loader,
            epoch=epoch)
        gc.collect()
        score = eval_epoch(
            model=model,
            data_loader=eval_data_loader,
            epoch=epoch)
        gc.collect()

        scheduler.step_epoch()
        scheduler.step_score(score)

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(args.experiment_path, 'model.pth'))


def main():
    utils.seed_everything(config.seed)
    lr = config.opt.lr
    train(lr)


if __name__ == '__main__':
    main()
