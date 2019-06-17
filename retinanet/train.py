import argparse
import gc
import itertools
import math
import os
import shutil

import numpy as np
import torch
import torch.distributions
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageDraw
from tensorboardX import SummaryWriter
from tqdm import tqdm

import lr_scheduler_wrapper
import utils
from config import Config
from optim import AdamW
from retinanet.dataset import Dataset, NUM_CLASSES
from retinanet.model import RetinaNet
from retinanet.transform import Resize, ToTensor, Normalize, BuildLabels, build_anchors_maps

# TODO: try largest lr before diverging
# TODO: check all plots rendered


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# TODO: check
class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, sampler):
        super().__init__(sampler)

        self.sampler = sampler

    def __len__(self):
        return len(self.sampler) // 2

    def __iter__(self):
        for i in self.sampler:
            if i % 2 == 1:
                continue

            yield [i, i + 1]


def compute_anchor(size, ratio, scale):
    h = math.sqrt(size**2 / ratio) * scale
    w = h * ratio

    return h, w


anchor_types = list(itertools.product([1 / 2, 1, 2 / 1], [2**0, 2**(1 / 3), 2**(2 / 3)]))
ANCHORS = [
    [compute_anchor(32, ratio, scale) for ratio, scale in anchor_types],
    [compute_anchor(64, ratio, scale) for ratio, scale in anchor_types],
    [compute_anchor(128, ratio, scale) for ratio, scale in anchor_types],
    [compute_anchor(256, ratio, scale) for ratio, scale in anchor_types],
    [compute_anchor(512, ratio, scale) for ratio, scale in anchor_types],
]

parser = argparse.ArgumentParser()
parser.add_argument('--config-path', type=str, required=True)
parser.add_argument('--experiment-path', type=str, default='./tf_log/retinanet')
parser.add_argument('--dataset-path', type=str, required=True)
parser.add_argument('--restore-path', type=str)
parser.add_argument('--workers', type=int, default=os.cpu_count())
args = parser.parse_args()
config = Config.from_yaml(args.config_path)
shutil.copy(args.config_path, utils.mkdir(args.experiment_path))

train_transform = T.Compose([
    Resize(600),
    ToTensor(),
    Normalize(mean=MEAN, std=STD),
    BuildLabels(ANCHORS)
])
eval_transform = T.Compose([
    Resize(600),
    ToTensor(),
    Normalize(mean=MEAN, std=STD),
    BuildLabels(ANCHORS)
])


def cat(a, b):
    image_a, (class_ids_a, boxes_a) = a
    image_b, (class_ids_b, boxes_b) = b

    image = torch.stack([image_a, image_b], 0)
    class_ids = torch.stack([class_ids_a, class_ids_b], 0)
    boxes = torch.stack([boxes_a, boxes_b], 0)

    return image, (class_ids, boxes)


def worker_init_fn(_):
    utils.seed_python(torch.initial_seed() % 2**32)


def focal_loss(input, target, gamma=2., alpha=0.25):
    norm = (target > 0).sum()

    target = torch.eye(NUM_CLASSES + 1).to(DEVICE)[target][:, 1:]

    prob = input.sigmoid()
    prob_true = prob * target + (1 - prob) * (1 - target)
    alpha = alpha * target + (1 - alpha) * (1 - target)
    weight = alpha * (1 - prob_true)**gamma

    loss = F.binary_cross_entropy_with_logits(input=input, target=target, reduction='none')
    loss = weight * loss
    loss = loss.sum() / norm

    return loss


# TODO: check loss
def compute_loss(input, target):
    input_class, input_regr = input
    target_class, target_regr = target

    class_mask = target_class != -1
    class_loss = focal_loss(input=input_class[class_mask], target=target_class[class_mask])
    assert class_loss.dim() == 0

    regr_mask = target_class > 0
    regr_loss = F.smooth_l1_loss(input=input_regr[regr_mask], target=target_regr[regr_mask])
    assert regr_loss.dim() == 0

    loss = class_loss + regr_loss

    return loss


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# TODO: stochastic weight averaging
# TODO: group images by buckets (size, ratio) and batch
# TODO: hinge loss clamp instead of minimum
# TODO: losses
# TODO: better one cycle
# TODO: cos vs lin
# TODO: load and restore state after lr finder
# TODO: shuffle split
# TODO: tune on large size
# TODO: cross val
# TODO: smart sampling
# TODO: cutout
# TODO: larger model
# TODO: load image as jpeg
# TODO: min 1 tag?
# TODO: speedup image loading
# TODO: smart sampling
# TODO: weight standartization
# TODO: build sched for lr find


def build_optimizer(optimizer, parameters, lr, beta, weight_decay):
    if optimizer == 'adam':
        return torch.optim.Adam(parameters, lr, betas=(beta, 0.999), weight_decay=weight_decay)
    elif optimizer == 'adamw':
        return AdamW(parameters, lr, betas=(beta, 0.999), weight_decay=weight_decay)
    elif optimizer == 'momentum':
        return torch.optim.SGD(parameters, lr, momentum=beta, weight_decay=weight_decay, nesterov=True)
    else:
        raise AssertionError('invalid OPT {}'.format(optimizer))


def decode_boxes(image_size, input):
    class_output, regr_output = input

    anchors = build_anchors_maps(image_size, ANCHORS).to(DEVICE)

    if class_output.dim() == 1:
        scores = torch.ones_like(class_output).float()
        class_ids = class_output - 1
        fg = class_output > 0
    else:
        scores, class_ids = class_output.max(1)
        fg = scores > 0.

    yx = regr_output[:, :2] * anchors[:, 2:] + anchors[:, :2]
    hw = regr_output[:, 2:].exp() * anchors[:, 2:]
    boxes = torch.cat([yx, hw], 1)

    boxes = boxes[fg]
    class_ids = class_ids[fg]
    scores = scores[fg]

    return boxes, class_ids, scores


def draw_boxes(image, boxes):
    device = image.device

    image = image.permute(1, 2, 0).data.cpu().numpy()
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for y, x, h, w in boxes.data.cpu().numpy():
        draw.rectangle(((x - w / 2, y - h / 2), (x + w / 2, y + h / 2)), outline=(0, 255, 0))
    image = torch.tensor(np.array(image) / 255).permute(2, 0, 1).to(device)

    return image


def train_epoch(model, optimizer, scheduler, data_loader, epoch):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'train'))

    metrics = {
        'loss': utils.Mean(),
    }

    model.train()
    for images, labels in tqdm(data_loader, desc='epoch {} train'.format(epoch)):
        images, labels = images.to(DEVICE), [l.to(DEVICE) for l in labels]
        logits = model(images)

        loss = compute_loss(input=logits, target=labels)
        metrics['loss'].update(loss.data.cpu().numpy())

        lr, _ = scheduler.get_lr()
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        # scheduler.step()

    with torch.no_grad():
        loss = metrics['loss'].compute_and_reset()

        image_size = images.size(2), images.size(3)
        mean, std = [torch.tensor(x).view(3, 1, 1).to(DEVICE) for x in [MEAN, STD]]
        boxes = [decode_boxes(image_size, (c, r))[0] for c, r in zip(*labels)]
        images_true = [draw_boxes(i * std + mean, b) for i, b in zip(images, boxes)]
        boxes = [decode_boxes(image_size, (c, r))[0] for c, r in zip(*logits)]
        images_pred = [draw_boxes(i * std + mean, b) for i, b in zip(images, boxes)]

        print('[EPOCH {}][TRAIN] loss: {:.4f}'.format(epoch, loss))
        writer.add_scalar('loss', loss, global_step=epoch)
        writer.add_scalar('learning_rate', lr, global_step=epoch)
        writer.add_image('images_true', torchvision.utils.make_grid(images_true, normalize=True), global_step=epoch)
        writer.add_image('images_pred', torchvision.utils.make_grid(images_pred, normalize=True), global_step=epoch)


def eval_epoch(model, data_loader, epoch):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'eval'))

    metrics = {
        'loss': utils.Mean(),
    }

    model.eval()
    with torch.no_grad():
        # predictions = []
        # targets = []

        for images, labels in tqdm(data_loader, desc='epoch {} evaluation'.format(epoch)):
            images, labels = images.to(DEVICE), [l.to(DEVICE) for l in labels]
            logits = model(images)

            # targets.append(labels)
            # predictions.append(logits)

            loss = compute_loss(input=logits, target=labels)
            metrics['loss'].update(loss.data.cpu().numpy())

        loss = metrics['loss'].compute_and_reset()

        # predictions = torch.cat(predictions, 0)
        # targets = torch.cat(targets, 0)
        score = epoch  # TODO:

        image_size = images.size(2), images.size(3)
        mean, std = [torch.tensor(x).view(3, 1, 1).to(DEVICE) for x in [MEAN, STD]]
        boxes = [decode_boxes(image_size, (c, r))[0] for c, r in zip(*labels)]
        images_true = [draw_boxes(i * std + mean, b) for i, b in zip(images, boxes)]
        boxes = [decode_boxes(image_size, (c, r))[0] for c, r in zip(*logits)]
        images_pred = [draw_boxes(i * std + mean, b) for i, b in zip(images, boxes)]

        print('[EPOCH {}][EVAL] loss: {:.4f}, score: {:.4f}'.format(epoch, loss, score))
        writer.add_scalar('loss', loss, global_step=epoch)
        writer.add_scalar('score', score, global_step=epoch)
        writer.add_image('images_true', torchvision.utils.make_grid(images_true, normalize=True), global_step=epoch)
        writer.add_image('images_pred', torchvision.utils.make_grid(images_pred, normalize=True), global_step=epoch)

        return score


def train():
    train_dataset = Dataset(args.dataset_path, train=True, transform=train_transform)
    train_dataset = torch.utils.data.Subset(train_dataset, list(range(1000)))
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=BatchSampler(torch.utils.data.RandomSampler(train_dataset)),
        num_workers=args.workers,
        worker_init_fn=worker_init_fn)

    eval_dataset = Dataset(args.dataset_path, train=False, transform=eval_transform)
    eval_dataset = torch.utils.data.Subset(eval_dataset, list(range(100)))
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_sampler=BatchSampler(torch.utils.data.SequentialSampler(eval_dataset)),
        num_workers=args.workers,
        worker_init_fn=worker_init_fn)

    model = RetinaNet(NUM_CLASSES, len(anchor_types))
    model = model.to(DEVICE)
    if args.restore_path is not None:
        model.load_state_dict(torch.load(args.restore_path))

    optimizer = build_optimizer(
        config.opt.type, model.parameters(), config.opt.lr, config.opt.beta, weight_decay=config.opt.weight_decay)

    if config.sched.type == 'multistep':
        scheduler = lr_scheduler_wrapper.EpochWrapper(
            torch.optim.lr_scheduler.MultiStepLR(optimizer, config.sched.multistep.steps))
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
        score = eval_epoch(
            model=model,
            data_loader=eval_data_loader,
            epoch=epoch)
        gc.collect()

        scheduler.step_epoch()
        scheduler.step_score(score)

        torch.save(model.state_dict(), os.path.join(args.experiment_path, 'model.pth'))


def main():
    utils.seed_python(config.seed)
    utils.seed_torch(config.seed)
    train()


if __name__ == '__main__':
    main()
