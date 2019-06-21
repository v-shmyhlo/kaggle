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
from PIL import Image, ImageDraw, ImageFont
from tensorboardX import SummaryWriter
from tqdm import tqdm

import lr_scheduler_wrapper
import utils
from config import Config
from detection.anchors import build_anchors_maps
from detection.dataset import Dataset, NUM_CLASSES
from detection.model import RetinaNet
from detection.transform import Resize, ToTensor, Normalize, BuildLabels, RandomCrop, RandomFlipLeftRight, \
    MaskCropAndResize, denormalize
from detection.utils import decode_boxes, boxes_yxhw_to_tlbr
from optim import AdamW

# TODO: visualization scores sigmoid

COLORS = np.random.uniform(51, 255, size=(NUM_CLASSES, 3)).round().astype(np.uint8)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MIN_IOU = 0.4
MAX_IOU = 0.5


def compute_anchor(size, ratio, scale):
    h = math.sqrt(size**2 / ratio) * scale
    w = h * ratio

    return h, w


anchor_types = list(itertools.product([1 / 2, 1, 2 / 1], [2**0, 2**(1 / 3), 2**(2 / 3)]))
ANCHORS = [
    None,
    [compute_anchor(32, ratio, scale) for ratio, scale in anchor_types],
    [compute_anchor(64, ratio, scale) for ratio, scale in anchor_types],
    [compute_anchor(128, ratio, scale) for ratio, scale in anchor_types],
    [compute_anchor(256, ratio, scale) for ratio, scale in anchor_types],
    [compute_anchor(512, ratio, scale) for ratio, scale in anchor_types],
]
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
anchor_maps = build_anchors_maps((config.image_size, config.image_size), ANCHORS, p2=False, p7=True).to(DEVICE)

train_transform = T.Compose([
    Resize(config.image_size),
    RandomCrop(config.image_size),
    RandomFlipLeftRight(),
    MaskCropAndResize(28),
    ToTensor(),
    Normalize(mean=MEAN, std=STD),
    BuildLabels(ANCHORS, p2=False, p7=True, min_iou=MIN_IOU, max_iou=MAX_IOU),
])
eval_transform = T.Compose([
    Resize(config.image_size),
    RandomCrop(config.image_size),
    MaskCropAndResize(28),
    ToTensor(),
    Normalize(mean=MEAN, std=STD),
    BuildLabels(ANCHORS, p2=False, p7=True, min_iou=MIN_IOU, max_iou=MAX_IOU),
])


class RandomSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, size):
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return min(self.size, len(self.dataset))

    def __getitem__(self, item):
        return self.dataset[np.random.randint(len(self.dataset))]


def worker_init_fn(_):
    utils.seed_python(torch.initial_seed() % 2**32)


def focal_loss(input, target, gamma=2., alpha=0.25):
    norm = (target > 0).sum()
    assert norm > 0

    target = utils.one_hot(target + 1, NUM_CLASSES + 2)[:, 2:]

    prob = input.sigmoid()
    prob_true = prob * target + (1 - prob) * (1 - target)
    alpha = alpha * target + (1 - alpha) * (1 - target)
    weight = alpha * (1 - prob_true)**gamma

    loss = F.binary_cross_entropy_with_logits(input=input, target=target, reduction='none')
    loss = (weight * loss).sum() / norm

    return loss


def smooth_l1_loss(input, target):
    assert input.numel() == target.numel()
    assert input.numel() > 0

    loss = F.smooth_l1_loss(input=input, target=target, reduction='mean')

    return loss


# TODO: check loss
def compute_loss(input, target):
    input_class, input_regr = input
    target_class, target_regr = target

    # classification loss
    class_mask = target_class != -1
    class_loss = focal_loss(input=input_class[class_mask], target=target_class[class_mask])
    assert class_loss.dim() == 0

    # regression loss
    regr_mask = target_class > 0
    regr_loss = smooth_l1_loss(input=input_regr[regr_mask], target=target_regr[regr_mask])
    assert regr_loss.dim() == 0

    loss = class_loss + regr_loss

    return loss


def build_optimizer(optimizer, parameters, lr, beta, weight_decay):
    if optimizer == 'adam':
        return torch.optim.Adam(parameters, lr, betas=(beta, 0.999), weight_decay=weight_decay)
    elif optimizer == 'adamw':
        return AdamW(parameters, lr, betas=(beta, 0.999), weight_decay=weight_decay)
    elif optimizer == 'momentum':
        return torch.optim.SGD(parameters, lr, momentum=beta, weight_decay=weight_decay, nesterov=True)
    else:
        raise AssertionError('invalid OPT {}'.format(optimizer))


def draw_boxes(image, detections, class_names, colors=COLORS):
    font = ImageFont.truetype('./imet/Droid+Sans+Mono+Awesome.ttf', size=14)

    class_ids, boxes, scores = detections
    boxes = boxes_yxhw_to_tlbr(boxes)

    device = image.device
    image = image.permute(1, 2, 0).data.cpu().numpy()
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    for c, (t, l, b, r), s in zip(class_ids.data.cpu().numpy(), boxes.data.cpu().numpy(), scores.data.cpu().numpy()):
        color = tuple(colors[c])
        text = '{}: {:.2f}'.format(class_names[c], s)
        size = draw.textsize(text, font=font)
        draw.rectangle(((l, t - size[1]), (l + size[0], t)), fill=color)
        draw.text((l, t - size[1]), text, font=font, fill=(0, 0, 0))
        draw.rectangle(((l, t), (r, b)), outline=color)

    image = torch.tensor(np.array(image) / 255).permute(2, 0, 1).to(device)

    return image


def compute_mask_loss(input, target):
    class_ids, boxes, masks, image_ids = target

    input = input[torch.arange(input.size(0)), class_ids]
    input = input.unsqueeze(1)

    return F.binary_cross_entropy_with_logits(input=input, target=masks, reduction='mean')


def train_epoch(model, optimizer, scheduler, data_loader, class_names, epoch):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'train'))

    metrics = {
        'loss': utils.Mean(),
    }

    model.train()
    for images, dets, maps in tqdm(data_loader, desc='epoch {} train'.format(epoch)):
        images, dets, maps = images.to(DEVICE), [d.to(DEVICE) for d in dets], [m.to(DEVICE) for m in maps]
        fpn_output, logits = model(images)
        mask_logits = model.roi_align_mask_head(fpn_output, dets)

        loss = compute_loss(input=logits, target=maps) + compute_mask_loss(input=mask_logits, target=dets)
        metrics['loss'].update(loss.data.cpu().numpy())

        lr, _ = scheduler.get_lr()
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        loss = metrics['loss'].compute_and_reset()

        dets = [decode_boxes((utils.one_hot(c + 1, NUM_CLASSES + 2)[:, 2:], r), anchor_maps) for c, r in zip(*maps)]
        images_true = [draw_boxes(denormalize(i, mean=MEAN, std=STD), d, class_names) for i, d in zip(images, dets)]
        dets = [decode_boxes((c, r), anchor_maps) for c, r in zip(*logits)]
        images_pred = [draw_boxes(denormalize(i, mean=MEAN, std=STD), d, class_names) for i, d in zip(images, dets)]

        print('[EPOCH {}][TRAIN] loss: {:.4f}'.format(epoch, loss))
        writer.add_scalar('loss', loss, global_step=epoch)
        writer.add_scalar('learning_rate', lr, global_step=epoch)
        writer.add_image(
            'images_true', torchvision.utils.make_grid(images_true, nrow=4, normalize=True), global_step=epoch)
        writer.add_image(
            'images_pred', torchvision.utils.make_grid(images_pred, nrow=4, normalize=True), global_step=epoch)


def eval_epoch(model, data_loader, class_names, epoch):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'eval'))

    metrics = {
        'loss': utils.Mean(),
    }

    model.eval()
    with torch.no_grad():
        for images, dets, maps in tqdm(data_loader, desc='epoch {} evaluation'.format(epoch)):
            images, dets, maps = images.to(DEVICE), [d.to(DEVICE) for d in dets], [m.to(DEVICE) for m in maps]
            fpn_output, logits = model(images)
            mask_logits = model.roi_align_mask_head(fpn_output, dets)

            loss = compute_loss(input=logits, target=maps) + compute_mask_loss(input=mask_logits, target=dets)
            metrics['loss'].update(loss.data.cpu().numpy())

        loss = metrics['loss'].compute_and_reset()
        score = 0  # TODO:

        dets = [decode_boxes((utils.one_hot(c + 1, NUM_CLASSES + 2)[:, 2:], r), anchor_maps) for c, r in zip(*maps)]
        images_true = [draw_boxes(denormalize(i, mean=MEAN, std=STD), d, class_names) for i, d in zip(images, dets)]
        dets = [decode_boxes((c, r), anchor_maps) for c, r in zip(*logits)]
        images_pred = [draw_boxes(denormalize(i, mean=MEAN, std=STD), d, class_names) for i, d in zip(images, dets)]

        print('[EPOCH {}][EVAL] loss: {:.4f}, score: {:.4f}'.format(epoch, loss, score))
        writer.add_scalar('loss', loss, global_step=epoch)
        writer.add_scalar('score', score, global_step=epoch)
        writer.add_image(
            'images_true', torchvision.utils.make_grid(images_true, nrow=4, normalize=True), global_step=epoch)
        writer.add_image(
            'images_pred', torchvision.utils.make_grid(images_pred, nrow=4, normalize=True), global_step=epoch)

        return score


def collate_cat_fn(batch):
    class_ids, boxes, masks = zip(*batch)
    image_ids = [torch.full_like(c, i) for i, c in enumerate(class_ids)]

    class_ids = torch.cat(class_ids, 0)
    boxes = torch.cat(boxes, 0)
    masks = torch.cat(masks, 0)
    image_ids = torch.cat(image_ids, 0)

    return class_ids, boxes, masks, image_ids


def collate_fn(batch):
    images, dets, maps = zip(*batch)

    images = torch.utils.data.dataloader.default_collate(images)
    dets = collate_cat_fn(dets)
    maps = torch.utils.data.dataloader.default_collate(maps)

    return images, dets, maps


def train():
    train_dataset = Dataset(args.dataset_path, train=True, transform=train_transform)
    class_names = train_dataset.class_names
    train_dataset = RandomSubset(train_dataset, config.train_size)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn)

    eval_dataset = Dataset(args.dataset_path, train=False, transform=eval_transform)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        num_workers=args.workers,
        collate_fn=collate_fn,
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
            class_names=class_names,
            epoch=epoch)
        gc.collect()
        score = eval_epoch(
            model=model,
            data_loader=eval_data_loader,
            class_names=class_names,
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
