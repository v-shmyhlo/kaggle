import os
from functools import partial

import click
import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm

from .dataset import EPS, TrainEvalDataset, load_train_eval_data
from .transform import LoadSignal, ToSpectrogram
from .utils import collate_fn


@click.command()
@click.option('--dataset-path', type=click.Path(exists=True), required=True)
def main(**kwargs):
    mean, std = np.load(os.path.join(kwargs['dataset_path'], 'stats.npy'))
    pad_value = (np.log(EPS) - mean) / std

    train_eval_data = load_train_eval_data(kwargs['dataset_path'], 'train_curated')
    train_noisy_data = load_train_eval_data(kwargs['dataset_path'], 'train_noisy')

    transform = T.Compose([
        LoadSignal(),
        ToSpectrogram(eps=EPS),
        T.ToTensor(),
        T.Normalize(mean=[mean], std=[std])
    ])

    dataset = torch.utils.data.ConcatDataset([
        TrainEvalDataset(train_eval_data, transform=transform),
        TrainEvalDataset(train_noisy_data, transform=transform)
    ])

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=120,
        drop_last=True,
        shuffle=True,
        num_workers=os.cpu_count(),
        collate_fn=partial(collate_fn, pad_value=pad_value))  # TODO: all args

    for _ in tqdm(data_loader):
        pass


if __name__ == '__main__':
    main()
