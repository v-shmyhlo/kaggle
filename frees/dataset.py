import pandas as pd
import argparse
from tqdm import tqdm
import os
import torch.utils.data
import numpy as np
import librosa
import soundfile
from PIL import Image

NUM_CLASSES = 80
EPS = 1e-7


class TrainEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data.iloc[i]

        image = load_image(row['fname'])
        if self.transform is not None:
            image = self.transform(image)

        label = np.zeros(NUM_CLASSES, dtype=np.float32)
        label[row['labels']] = 1.

        id = os.path.split(row['fname'])[-1]

        return image, label, id


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data.iloc[i]

        image = load_image(row['fname'])
        if self.transform is not None:
            image = self.transform(image)

        id = os.path.split(row['fname'])[-1]

        return image, id


def load_image(path):
    sig, rate = soundfile.read(path, dtype=np.float32)

    n_fft = round(0.025 * rate)  # TODO: refactor
    hop_length = round(0.01 * rate)  # TODO: refactor

    x = librosa.core.stft(sig, n_fft=n_fft, hop_length=hop_length)
    x = np.abs(x)
    x = np.dot(librosa.filters.mel(rate, n_fft), x)
    x = np.log(x + EPS)  # TODO: add eps, or clip?

    x = Image.fromarray(x)

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    id_to_class = list(pd.read_csv(os.path.join(args.dataset_path, 'sample_submission.csv')).columns[1:])
    class_to_id = {c: i for i, c in enumerate(id_to_class)}

    train_eval_data = pd.read_csv(os.path.join(args.dataset_path, 'train_curated.csv'))
    train_eval_data['fname'] = train_eval_data['fname'].apply(
        lambda x: os.path.join(args.dataset_path, 'train_curated', x))
    train_eval_data['labels'] = train_eval_data['labels'].apply(lambda x: [class_to_id[c] for c in x.split(',')])

    test_data = pd.DataFrame({'fname': os.listdir(os.path.join(args.dataset_path, 'test'))})
    test_data['fname'] = test_data['fname'].apply(
        lambda x: os.path.join(args.dataset_path, 'train_curated', x))

    if args.debug:
        train_eval_data['fname'] = './frees/sample.wav'
        test_data['fname'] = './frees/sample.wav'

    train_eval_dataset = TrainEvalDataset(train_eval_data)
    num_features = 128

    # compute mean
    global_mean = 0.
    feature_mean = np.zeros((num_features, 1))
    size = 0
    for image, _, _ in tqdm(train_eval_dataset):
        image = np.array(image)
        global_mean += image.sum()
        feature_mean += image.sum(1, keepdims=True)
        size += image.shape[1]
    global_mean = global_mean / (size * num_features)
    feature_mean = feature_mean / size

    # compute std
    global_std = 0.
    feature_std = np.zeros((num_features, 1))
    size = 0
    for image, _, _ in tqdm(train_eval_dataset):
        image = np.array(image)
        global_std += ((image - global_mean)**2).sum()
        feature_std += ((image - feature_mean)**2).sum(1, keepdims=True)
        size += image.shape[1]
    global_std = np.sqrt(global_std / (size * num_features))
    feature_std = np.sqrt(feature_std / size)

    stats = (
        (global_mean.squeeze().astype(np.float32), global_std.squeeze().astype(np.float32)),
        (feature_mean.squeeze().astype(np.float32), feature_std.squeeze().astype(np.float32)))

    np.save(os.path.join(args.dataset_path, 'stats.npy'), stats)


if __name__ == '__main__':
    main()
