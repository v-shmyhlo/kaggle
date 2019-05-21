import click
import utils
import pickle
from multiprocessing import Pool
from tqdm import tqdm
import os
import torch
from .dataset import EPS, TrainEvalDataset, load_train_eval_data
from .transform import LoadSignal, ToSpectrogram  # , AugmentSignal
import torchvision.transforms as T

transform = T.Compose([
    LoadSignal(),
    ToSpectrogram(eps=EPS),
])

transform_aug = T.Compose([
    LoadSignal(),
    # AugmentSignal(),
    ToSpectrogram(eps=EPS),
])


def preprocess(input):
    row, _, _ = input

    utils.mkdir(os.path.dirname(row['spectra_path']))
    with open(row['spectra_path'], 'wb') as f:
        pickle.dump(transform(row), f)

    utils.mkdir(os.path.dirname(row['spectra_aug_path']))
    with open(row['spectra_aug_path'], 'wb') as f:
        pickle.dump(transform_aug(row), f)


@click.command()
@click.option('--dataset-path', type=click.Path(exists=True), required=True)
@click.option('--workers', type=int, default=os.cpu_count())
def main(**kwargs):
    # mean, std = np.load(os.path.join(kwargs['dataset_path'], 'stats.npy'))
    # pad_value = (np.log(EPS) - mean) / std

    train_eval_data = load_train_eval_data(kwargs['dataset_path'], 'train_curated')
    train_noisy_data = load_train_eval_data(kwargs['dataset_path'], 'train_noisy')

    dataset = torch.utils.data.ConcatDataset([
        TrainEvalDataset(train_eval_data),
        TrainEvalDataset(train_noisy_data)
    ])

    with Pool(kwargs['workers']) as pool:
        pool.map(preprocess, tqdm(dataset))


if __name__ == '__main__':
    main()
