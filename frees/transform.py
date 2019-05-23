import soundfile
import numpy as np
import torch


# from pysndfx import AudioEffectsChain


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
    def __init__(self, splits):
        self.splits = splits

    def __call__(self, input):
        size, = input.shape
        window = size // self.splits

        chunks = [input[i * window:(i + 1) * window] for i in range(self.splits)]
        np.random.shuffle(chunks)
        input = np.concatenate(chunks, 0)

        return input


# class RandomSplitConcat(object):
#     def __init__(self, splits):
#         self.splits = splits
#
#     def __call__(self, input, recur=True):
#         size, = input.shape
#
#         i = np.random.randint(size // 8, size - size // 8)
#         left, right = np.split(input, [i], 0)
#         if recur:
#             left, right = self(left, recur=False), self(right, recur=False)
#         input = np.concatenate([right, left], 0)
#
#         return input


class Cutout(object):
    def __init__(self, fraction):
        self.fraction = fraction

    def __call__(self, input):
        size, = input.shape
        window = round(size * self.fraction)

        start = np.random.randint(-window, size)
        input[np.clip(start, 0, size):np.clip(start + window, 0, size)] = 0.

        return input

# class AugmentSignal(object):
#     def __call__(self, input):
#         # effect = AudioEffectsChain()
#         # effect = effect.pitch(np.random.uniform(-100, 100))
#         # effect = effect.tempo(np.random.uniform(0.8, 1.2))
#
#         # return effect(input)
#
#         return input

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
