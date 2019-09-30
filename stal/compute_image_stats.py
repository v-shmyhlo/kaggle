import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from stal.dataset import TrainEvalDataset


def one_hot(input, num_classes):
    return np.eye(num_classes)[input]


def main():
    dataset_path = '../../data/stal'

    train_eval_data = pd.read_csv(os.path.join(dataset_path, 'train.csv'), converters={'EncodedPixels': str})
    train_eval_data['root'] = os.path.join(dataset_path, 'train_images')
    train_eval_dataset = TrainEvalDataset(train_eval_data)

    sizes = []
    rs = []
    for input in tqdm(train_eval_dataset):
        sizes.append(input['image'].size)
        rs.append(one_hot(np.array(input['mask']), 5).mean((0, 1)))

    rs = np.array(rs)
    print(set(sizes))
    print(rs.shape)
    print(rs.min(0), rs.max(0))


if __name__ == '__main__':
    main()
