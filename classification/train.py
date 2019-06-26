import argparse
import gc
import os
import shutil

import torch
import torch.distributions
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torchvision
import torchvision.datasets
import torchvision.transforms as T
from tensorboardX import SummaryWriter
from tqdm import tqdm

import lr_scheduler_wrapper
import utils
from classification.mobilenet_v3 import MobileNetV3
from config import Config

NUM_CLASSES = 1000
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--config-path', type=str, required=True)
parser.add_argument('--experiment-path', type=str, default='./tf_log/detection')
parser.add_argument('--dataset-path', type=str, required=True)
parser.add_argument('--restore-path', type=str)
parser.add_argument('--workers', type=int, default=os.cpu_count())
args = parser.parse_args()
config = Config.from_yaml(args.config_path)
shutil.copy(args.config_path, utils.mkdir(args.experiment_path))

train_transform = T.Compose([
    T.RandomSizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),

    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])
eval_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),

    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


def worker_init_fn(_):
    utils.seed_python(torch.initial_seed() % 2**32)


def compute_loss(input, target):
    print(input.shape, target.shape)
    loss = F.cross_entropy(input=input, target=target)
    print(loss)
    fail

    return loss


def compute_metric(input, target):
    print(input.shape, target.shape)
    accuracy = input.argmax() == target
    print(accuracy)
    fail

    return accuracy


def build_optimizer(optimizer, parameters, lr, beta, weight_decay):
    if optimizer == 'momentum':
        return torch.optim.SGD(parameters, lr, momentum=beta, weight_decay=weight_decay, nesterov=True)
    elif optimizer == 'rmsprop':
        return torch.optim.RMSProp(parameters, lr, momentum=beta, weight_decay=weight_decay, nesterov=True)
    elif optimizer == 'adam':
        return torch.optim.Adam(parameters, lr, betas=(beta, 0.999), weight_decay=weight_decay)
    else:
        raise AssertionError('invalid OPT {}'.format(optimizer))


def train_epoch(model, optimizer, scheduler, data_loader, epoch):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'train'))

    metrics = {
        'loss': utils.Mean(),
    }

    model.train()
    for images, labels in tqdm(data_loader, desc='epoch {} train'.format(epoch)):
        images, labels = images.to(DEVICE), [m.to(DEVICE) for m in labels]
        logits = model(images)

        loss = compute_loss(input=logits, target=labels)
        metrics['loss'].update(loss.data.cpu().numpy())

        lr, _ = scheduler.get_lr()
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        loss = metrics['loss'].compute_and_reset()

        print('[EPOCH {}][TRAIN] loss: {:.4f}'.format(epoch, loss))
        writer.add_scalar('loss', loss, global_step=epoch)
        writer.add_scalar('learning_rate', lr, global_step=epoch)
        writer.add_image('images', torchvision.utils.make_grid(images, nrow=4, normalize=True), global_step=epoch)


def eval_epoch(model, data_loader, epoch):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'eval'))

    metrics = {
        'loss': utils.Mean(),
    }

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='epoch {} evaluation'.format(epoch)):
            images, labels = images.to(DEVICE), [m.to(DEVICE) for m in labels]
            logits = model(images)

            loss = compute_loss(input=logits, target=labels)
            metrics['loss'].update(loss.data.cpu().numpy())

            metric = compute_metric(input=logits, target=labels)
            metrics['metric'].update(metric.data.cpu().numpy())

        loss = metrics['loss'].compute_and_reset()
        metric = metrics['metric'].compute_and_reset()

        print('[EPOCH {}][EVAL] loss: {:.4f}, metric: {:.4f}'.format(epoch, loss, metric))
        writer.add_scalar('loss', loss, global_step=epoch)
        writer.add_scalar('metric', metric, global_step=epoch)
        writer.add_image('images', torchvision.utils.make_grid(images, nrow=4, normalize=True), global_step=epoch)

        return metric


def train():
    # FIXME:
    train_dataset = torchvision.datasets.ImageNet(
        args.dataset_path, split='val', download=True, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=args.workers,
        worker_init_fn=worker_init_fn)

    eval_dataset = torchvision.datasets.ImageNet(
        args.dataset_path, split='val', download=True, transform=eval_transform)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        num_workers=args.workers,
        worker_init_fn=worker_init_fn)

    model = MobileNetV3(3, NUM_CLASSES)
    model = model.to(DEVICE)
    if args.restore_path is not None:
        model.load_state_dict(torch.load(args.restore_path))

    optimizer = build_optimizer(
        config.opt.type, model.parameters(), config.opt.lr, config.opt.beta, weight_decay=config.opt.weight_decay)

    if config.sched.type == 'multistep':
        scheduler = lr_scheduler_wrapper.EpochWrapper(
            torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.sched.step_size, gamma=config.sched.step.decay))
    else:
        raise AssertionError('invalid sched {}'.format(config.sched.type))

    for epoch in range(config.epochs):
        train_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data_loader=train_data_loader,
            epoch=epoch)
        gc.collect()
        metric = eval_epoch(
            model=model,
            data_loader=eval_data_loader,
            epoch=epoch)
        gc.collect()

        scheduler.step_epoch()
        scheduler.step_score(metric)

        torch.save(model.state_dict(), os.path.join(args.experiment_path, 'model.pth'))


def main():
    utils.seed_python(config.seed)
    utils.seed_torch(config.seed)
    train()


if __name__ == '__main__':
    main()
