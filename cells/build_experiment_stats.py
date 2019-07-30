import gc
import os
import resource

import click
import pandas as pd
import torch
import torch.utils.data
import torchvision.transforms as T
from tqdm import tqdm

from cells.dataset import TestDataset
from cells.transforms import ApplyTo, SplitInSites, Extract, ToTensor

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


@click.command()
@click.option('--dataset-path', type=click.Path(), required=True)
@click.option('--workers', type=click.INT, default=os.cpu_count())
def main(dataset_path, workers):
    transform = T.Compose([
        ApplyTo(
            ['image'],
            T.Compose([
                SplitInSites(),
                T.Lambda(lambda xs: torch.stack([ToTensor()(x) for x in xs], 0)),
            ])),
        Extract(['image']),
    ])

    train_data = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    train_data['root'] = os.path.join(dataset_path, 'train')
    test_data = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    test_data['root'] = os.path.join(dataset_path, 'test')
    data = pd.concat([train_data, test_data])

    stats = {}
    for exp, group in tqdm(data.groupby('experiment')):
        dataset = TestDataset(group, transform=transform)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            num_workers=workers)

        with torch.no_grad():
            images = [images for images, in data_loader]
            images = torch.cat(images, 0)
            mean = images.mean((0, 1, 3, 4))
            std = images.std((0, 1, 3, 4))
            stats[exp] = mean, std

            del images, mean, std
            gc.collect()

    torch.save(stats, 'experiment_stats.pth')


if __name__ == '__main__':
    main()
