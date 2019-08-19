import argparse
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import torch
import torch.distributions
import torch.utils
import torch.utils.data
import torchvision.transforms as T
from tqdm import tqdm

import utils
from cells.dataset import NUM_CLASSES, TestDataset
from cells.model import Model
from cells.transforms import Extract, ApplyTo, Resize, ToTensor, RandomSite, SplitInSites, \
    RandomCrop, CenterCrop, NormalizeByExperimentStats, NormalizeByPlateStats, Resetable
from config import Config

FOLDS = list(range(1, 3 + 1))
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--config-path', type=str, required=True)
parser.add_argument('--experiment-path', type=str, default='./tf_log/cells')
parser.add_argument('--dataset-path', type=str, required=True)
parser.add_argument('--workers', type=int, default=os.cpu_count())
parser.add_argument('--fold', type=int, choices=FOLDS)
args = parser.parse_args()
config = Config.from_yaml(args.config_path)
shutil.copy(args.config_path, utils.mkdir(args.experiment_path))
assert config.resize_size == config.crop_size


class RandomResize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, input):
        size = round(np.random.uniform(self.min_size, self.max_size))
        input = Resize(size)(input)

        return input


random_crop = Resetable(RandomCrop)
center_crop = Resetable(CenterCrop)
infer_image_transform = Resetable(lambda tta: test_image_transform if tta else eval_image_transform)
to_tensor = ToTensor()

if config.normalize is None:
    normalize = T.Compose([])
elif config.normalize == 'experiment':
    normalize = NormalizeByExperimentStats(
        torch.load('./experiment_stats.pth'))  # TODO: needs realtime computation on private
elif config.normalize == 'plate':
    normalize = NormalizeByPlateStats(
        torch.load('./plate_stats.pth'))  # TODO: needs realtime computation on private
else:
    raise AssertionError('invalide normalization {}'.format(config.normalize))

eval_image_transform = T.Compose([
    RandomSite(),
    Resize(config.resize_size),
    center_crop,
    to_tensor,
])
test_image_transform = T.Compose([
    Resize(config.resize_size),
    center_crop,
    SplitInSites(),
    T.Lambda(lambda xs: torch.stack([to_tensor(x) for x in xs], 0)),
])
test_transform = T.Compose([
    ApplyTo(
        ['image'],
        infer_image_transform),
    normalize,
    Extract(['image', 'feat', 'exp', 'id']),
])


def update_transforms(p):
    if not config.progressive_resize:
        p = 1.

    assert 0. <= p <= 1.

    crop_size = round(224 + (config.crop_size - 224) * p)
    print('update transforms p: {:.2f}, crop_size: {}'.format(p, crop_size))
    random_crop.reset(crop_size)
    center_crop.reset(crop_size)


def worker_init_fn(_):
    utils.seed_python(torch.initial_seed() % 2**32)


def compute_features(folds, data):
    with torch.no_grad():
        for fold in folds:
            fold_embds, _, fold_ids = compute_features_using_fold(fold, data)
            print(fold_embds.shape)
            fold_embds = fold_embds.data.cpu().numpy()
            fold_embds = {id: emb for id, emb in zip(fold_ids, fold_embds)}

            with open('embeddings_{}.pkl'.format(fold), 'wb') as f:
                pickle.dump(fold_embds, f)


def compute_features_using_fold(fold, data):
    dataset = TestDataset(data, transform=test_transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size // 2,
        num_workers=args.workers,
        worker_init_fn=worker_init_fn)

    model = Model(config.model, NUM_CLASSES, return_features=True)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(args.experiment_path, 'model_{}.pth'.format(fold))))

    model.eval()
    with torch.no_grad():
        fold_embs = []
        fold_exps = []
        fold_ids = []

        for images, feats, exps, ids in tqdm(data_loader, desc='fold {} inference'.format(fold)):
            images, feats = images.to(DEVICE), feats.to(DEVICE)

            b, n, c, h, w = images.size()
            images = images.view(b * n, c, h, w)
            feats = feats.view(b, 1, 2).repeat(1, n, 1).view(b * n, 2)
            _, embds = model(images, feats)
            embds = embds.view(b, n, embds.size(1))

            fold_embs.append(embds)
            fold_exps.extend(exps)
            fold_ids.extend(ids)

        fold_embs = torch.cat(fold_embs, 0)

    return fold_embs, fold_exps, fold_ids


def main():
    utils.seed_python(config.seed)
    utils.seed_torch(config.seed)

    train_eval_data = pd.read_csv(os.path.join(args.dataset_path, 'train.csv'))
    train_eval_data['root'] = os.path.join(args.dataset_path, 'train')

    test_data = pd.read_csv(os.path.join(args.dataset_path, 'test.csv'))
    test_data['root'] = os.path.join(args.dataset_path, 'test')

    data = pd.concat([train_eval_data, test_data])

    if args.fold is None:
        folds = FOLDS
    else:
        folds = [args.fold]

    infer_image_transform.reset(tta=True)
    update_transforms(1.)  # FIXME:
    compute_features(folds, data)


if __name__ == '__main__':
    main()
