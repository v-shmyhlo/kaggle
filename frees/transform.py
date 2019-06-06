import soundfile
import numpy as np
import itertools
import torch
from pysndfx import AudioEffectsChain


class LoadSignal(object):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def __call__(self, input):
        # TODO: soundfile vs librosa
        sig, rate = soundfile.read(input['path'], dtype=np.float32)
        assert rate == self.sample_rate

        return sig


class ToTensor(object):
    def __call__(self, input):
        return torch.tensor(input)


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, input):
        size, = input.shape

        if size < self.size:
            return input

        start = np.random.randint(0, size - self.size + 1)
        size = self.size

        input = input[start:start + size]

        return input


class RandomSplitConcat(object):
    def __init__(self, min_size):
        self.min_size = min_size

    def __call__(self, input):
        shape = input.shape

        splits = self.random_split(input)
        splits = itertools.chain.from_iterable(self.random_split(s) for s in splits)
        splits = list(splits)

        np.random.shuffle(splits)
        input = np.concatenate(splits)
        assert input.shape == shape

        return input

    def random_split(self, input):
        size, = input.shape

        if size < self.min_size * 2:
            return input,

        i = np.random.randint(self.min_size, size - self.min_size + 1)
        splits = np.split(input, [i])
        assert all(s.shape[0] >= self.min_size for s in splits)

        return splits


class Cutout(object):
    def __init__(self, fraction):
        self.fraction = fraction

    def __call__(self, input):
        size, = input.shape
        window = round(size * self.fraction)

        start = np.random.randint(-window, size)
        input[np.clip(start, 0, size):np.clip(start + window, 0, size)] = 0.

        return input


class AudioEffect(object):
    def __call__(self, input):
        effect = AudioEffectsChain()

        if np.random.uniform() > 0.5:
            effect = effect.pitch(np.random.uniform(-300, 300))
        if np.random.uniform() > 0.5:
            effect = effect.tempo(np.random.uniform(0.8, 1.2))
        if np.random.uniform() > 0.5:
            effect = effect.reverb(np.random.uniform(0, 100))

        return effect(input)

# class LoadSpectra(object):
#     def __init__(self, augmented=False):
#         self.augmented = augmented
#
#     def __call__(self, input):
#         if self.augmented:
#             path = input['spectra_aug_path']
#         else:
#             path = input['spectra_path']
#
#         with open(path, 'rb') as f:
#             return pickle.load(f)


# class ToSpectrogram(object):
#     def __init__(self, eps):
#         self.eps = eps
#
#     def __call__(self, input):
#         sig, rate = input
#
#         n_fft = round(0.025 * rate)
#         hop_length = round(0.01 * rate)
#
#         spectra = librosa.core.stft(sig, n_fft=n_fft, hop_length=hop_length)
#         spectra = np.abs(spectra)
#         spectra = np.dot(librosa.filters.mel(rate, n_fft), spectra)
#         spectra = np.log(spectra + self.eps)
#
#         spectra = Image.fromarray(spectra)
#
#         return spectra


# class CentralCrop(object):
#     def __init__(self, size):
#         self.size = size
#
#     def __call__(self, input):
#         w, h = input.size
#
#         if w < self.size:
#             return input
#
#         i = 0
#         j = (w - self.size) // 2
#         w = self.size
#
#         input = F.crop(input, i, j, h, w)
#
#         return input
