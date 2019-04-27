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

        return image, label, row['fname']


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

        return image, row['fname']


def load_image(path):
    sig, rate = soundfile.read(path, dtype=np.float32)

    n_fft = round(0.025 * rate)  # TODO: refactor
    hop_length = round(0.01 * rate)  # TODO: refactor

    x = librosa.core.stft(sig, n_fft=n_fft, hop_length=hop_length)
    x = np.abs(x)
    x = np.dot(librosa.filters.mel(rate, n_fft), x)
    x = np.log(x + EPS)  # TODO: add eps, or clip?

    # x = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
    x = Image.fromarray(x)

    return x
