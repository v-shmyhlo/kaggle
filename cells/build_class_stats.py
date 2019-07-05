import itertools
import os
import resource
from functools import partial
from multiprocessing import Pool

import click
import pandas as pd
import torch
import torch.utils.data
from tqdm import tqdm

from cells.dataset import NUM_CLASSES

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


# TODO: check
def build_class_stats(class_id, data, stats):
    ids = data['id_code'][data['sirna'] == class_id]
    class_stats = [
        stats[(stats['id_code'] == id) & (stats['site'] == s)]
        for id, s in itertools.product(ids, [1, 2])]
    assert all(all(s['channel'] == range(1, 7)) for s in class_stats)
    class_stats = [s[['mean', 'std']].values.T for s in class_stats]
    class_stats = torch.tensor(class_stats, dtype=torch.float)
    class_stats /= 255
    assert class_stats.size() == (len(ids) * 2, 2, 6)

    return class_stats


@click.command()
@click.option('--dataset-path', type=click.Path(), required=True)
@click.option('--workers', type=click.INT, default=os.cpu_count())
def main(dataset_path, workers):
    data = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    stats = pd.read_csv(os.path.join(dataset_path, 'pixel_stats.csv'))

    with Pool(workers) as pool:
        class_to_stats = pool.map(
            partial(build_class_stats, data=data, stats=stats),
            tqdm(range(NUM_CLASSES), desc='building class stats'))

    torch.save(class_to_stats, './stats.pth')


if __name__ == '__main__':
    main()
