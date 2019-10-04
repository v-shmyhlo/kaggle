import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from stal.dataset import TrainEvalDataset, NUM_CLASSES


def one_hot(input, num_classes):
    return np.eye(num_classes)[input]


def main():
    if os.path.exists('./stal/stats.npy'):
        areas = np.load('./stal/stats.npy')
    else:
        dataset_path = '../../data/stal'
        train_eval_data = pd.read_csv(os.path.join(dataset_path, 'train.csv'), converters={'EncodedPixels': str})
        train_eval_data['root'] = os.path.join(dataset_path, 'train_images')
        train_eval_dataset = TrainEvalDataset(train_eval_data)

        areas = []
        for input in tqdm(train_eval_dataset):
            mask = one_hot(np.array(input['mask']), NUM_CLASSES)[:, :, 1:]
            area = mask.sum()
            areas.append(area)
        areas = np.array(areas)
        np.save('./stal/stats.npy', areas)

    # indices = np.argsort(areas)
    # buckets = np.zeros(areas.shape, dtype=np.int)
    # n_buckets = 30
    #
    # for i in range(n_buckets):
    #     chunk_size = np.ceil(len(indices) / n_buckets).astype(np.int)
    #     s = indices[chunk_size * i:chunk_size * (i + 1)]
    #     buckets[s] = i
    #
    # print(np.bincount(buckets))
    # for i in range(n_buckets):
    #     print(i, areas[buckets == i].min(), areas[buckets == i].max())


if __name__ == '__main__':
    main()
