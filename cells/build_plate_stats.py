import os
from multiprocessing import Pool

import click
import pandas as pd

from cells.dataset import NUM_CLASSES


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
