import os

import pandas as pd
from tqdm import tqdm

from stal.dataset import TrainEvalDataset


def main():
    dataset_path = '../../data/stal'

    train_eval_data = pd.read_csv(os.path.join(dataset_path, 'train.csv'), converters={'EncodedPixels': str})
    train_eval_data['root'] = os.path.join(dataset_path, 'train_images')
    train_eval_dataset = TrainEvalDataset(train_eval_data)

    sizes = []
    for input in tqdm(train_eval_dataset):
        sizes.append(input['image'].size)
    print(set(sizes))


if __name__ == '__main__':
    main()
