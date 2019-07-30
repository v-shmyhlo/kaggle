import os
import pickle
import sys
from multiprocessing import Pool

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../rxrx1-utils')
import rxrx.io as rio


@click.command()
@click.option('--dataset-path', type=click.Path(), required=True)
@click.option('--workers', type=click.INT, default=os.cpu_count())
def main(dataset_path, workers):
    train_data = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    train_data['root'] = dataset_path

    mean_images = {}
    for exp, group in train_data.groupby('experiment'):
        rows = [row for i, row in group.iterrows()]
        with Pool(workers) as pool:
            images = pool.map(load_image, tqdm(rows, desc='loading {}'.format(exp)))
            images = np.concatenate(images, 0)
            mean_images[exp] = images.mean(0)

    with open('./mean_images.pkl', 'wb') as f:
        pickle.dump(mean_images, f)


def load_image(row):
    images = []
    for site in [1, 2]:
        image = rio.load_site('train', row['experiment'], row['plate'], row['well'], site, base_path=row['root'])
        images.append(image)

    images = np.stack(images, 0)

    return images


if __name__ == '__main__':
    main()
