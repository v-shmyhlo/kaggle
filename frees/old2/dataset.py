import pandas as pd
import argparse
from tqdm import tqdm
import os
import torch.utils.data
import numpy as np

ID_TO_CLASS = list(pd.read_csv(os.path.join(os.path.dirname(__file__), 'sample_submission.csv')).columns[1:])
CLASS_TO_ID = {c: i for i, c in enumerate(ID_TO_CLASS)}
NUM_CLASSES = len(ID_TO_CLASS)
EPS = 1e-7


class TrainEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data.iloc[i]

        image = row
        if self.transform is not None:
            image = self.transform(image)

        label = np.zeros(NUM_CLASSES, dtype=np.float32)
        label[row['labels']] = 1.

        return image, label, row['id']


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data.iloc[i]

        image = row
        if self.transform is not None:
            image = self.transform(image)

        return image, row['id']


def load_train_eval_data(path, name):
    data = pd.read_csv(os.path.join(path, '{}.csv'.format(name)))
    data = data.rename({'fname': 'id'}, axis='columns')
    data['path'] = data['id'].apply(
        lambda x: os.path.join(path, name, x))
    data['spectra_path'] = data['id'].apply(
        lambda x: os.path.join(path, '{}_spectra'.format(name), '{}.pkl'.format(x)))
    data['spectra_aug_path'] = data['id'].apply(
        lambda x: os.path.join(path, '{}_spectra_aug'.format(name), '{}.pkl'.format(x)))
    data['labels'] = data['labels'].apply(
        lambda x: [CLASS_TO_ID[c] for c in x.split(',')])

    return data


def load_test_data(path, name):
    data = pd.DataFrame({'id': os.listdir(os.path.join(path, name))})
    data['path'] = data['id'].apply(
        lambda x: os.path.join(path, name, x))
    data['spectra_path'] = data['id'].apply(
        lambda x: os.path.join(path, '{}_spectra'.format(name), '{}.pkl'.format(x)))

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True)
    args = parser.parse_args()

    train_eval_data = load_train_eval_data(args.dataset_path, 'train_curated')
    train_eval_dataset = TrainEvalDataset(train_eval_data)

    # compute mean
    mean = 0.
    size = 0
    for image, _, _ in tqdm(train_eval_dataset):
        image = np.array(image)
        mean += image.sum()
        size += image.size
    mean = mean / size

    # compute std
    std = 0.
    size = 0
    for image, _, _ in tqdm(train_eval_dataset):
        image = np.array(image)
        std += ((image - mean)**2).sum()
        size += image.size
    std = np.sqrt(std / size)

    # save stats
    stats = mean.squeeze().astype(np.float32), std.squeeze().astype(np.float32)
    np.save(os.path.join(args.dataset_path, 'stats.npy'), stats)


if __name__ == '__main__':
    main()
