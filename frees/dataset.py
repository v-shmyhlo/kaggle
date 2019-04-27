import torch.utils.data
import os
import numpy as np
import librosa
import soundfile

NUM_CLASSES = 80


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

        return image, label, row['fname']


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.data = os.listdir(os.path.join(args.dataset_path, 'test'))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path = self.data[i]
        id = os.path.splitext(path)[0]

        image = load_image(os.path.join(args.dataset_path, 'test', path))
        if self.transform is not None:
            image = self.transform(image)

        return image, id


def load_image(path):
    sig, rate = soundfile.read(path, dtype=np.float32)

    n_fft = round(0.025 * rate)  # TODO: refactor
    hop_length = round(0.01 * rate)  # TODO: refactor

    # print(rate)
    # print(n_fft, hop_length)
    # print(n_fft * (1 / rate), hop_length * (1 / rate))

    # x = librosa.feature.melspectrogram(sig, n_fft=n_fft, hop_length=hop_length)
    x = librosa.core.stft(sig, n_fft=n_fft, hop_length=hop_length)
    x = np.abs(x) + 1e-7  # TODO: add eps?
    x = np.log(x)
    x = np.dot(librosa.filters.mel(rate, n_fft), x)
    # x = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)

    return x
