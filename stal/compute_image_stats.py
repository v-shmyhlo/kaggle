import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from stal.dataset import TrainEvalDataset, NUM_CLASSES, build_data


def one_hot(input, num_classes):
    return np.eye(num_classes)[input]


def main():
    if os.path.exists('./stal/stats.npy'):
        all_areas = np.load('./stal/stats.npy')
    else:
        dataset_path = '../../data/stal'
        train_eval_data = pd.read_csv(os.path.join(dataset_path, 'train.csv'), converters={'EncodedPixels': str})
        train_eval_data['root'] = os.path.join(dataset_path, 'train_images')
        train_eval_data = build_data(train_eval_data)
        train_eval_dataset = TrainEvalDataset(train_eval_data)

        all_areas = []
        for input in tqdm(train_eval_dataset):
            mask = one_hot(np.array(input['mask']), NUM_CLASSES)[:, :, 1:]
            area = mask.sum((0, 1))
            all_areas.append(area)
        all_areas = np.array(all_areas)
        np.save('./stal/stats.npy', all_areas)

    all_buckets = compute_buckets(all_areas, num_buckets=5)
    print(all_buckets.shape)
    x = np.eye(5 + 1)[all_buckets]
    print(x.shape)
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    print(x.shape)


def compute_buckets(all_areas, num_buckets, debug=False):
    print('num_buckets: {}'.format(num_buckets))
   
    all_buckets = []
    for c in range(all_areas.shape[1]):
        areas = all_areas[:, c]
        buckets = np.zeros(areas.shape, dtype=np.int)

        indices = np.argsort(areas)
        indices = indices[areas[indices] > 0.]

        for i in range(num_buckets):
            chunk_size = np.ceil(len(indices) / num_buckets).astype(np.int)
            s = indices[chunk_size * i:chunk_size * (i + 1)]
            buckets[s] = i + 1
        all_buckets.append(buckets)

        if debug:
            print('class {}: {}'.format(c, np.bincount(buckets)))
            for i in range(num_buckets + 1):
                print('class {}, bucket {}: {} {}'.format(c, i, areas[buckets == i].min(), areas[buckets == i].max()))

    all_buckets = np.stack(all_buckets, 1)

    return all_buckets


if __name__ == '__main__':
    main()
