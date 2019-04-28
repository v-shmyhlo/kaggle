import numpy as np
import matplotlib.pyplot as plt
import soundfile
import scipy.signal
import librosa

librosa.power_to_db
librosa.feature.melspectrogram

sig, rate = soundfile.read('./frees/sample.wav', dtype=np.float32)
sig = sig[:rate * 3]

plt.plot(sig)
plt.show()

n_fft = round(0.025 * rate)  # TODO: refactor
hop_length = round(0.01 * rate)  # TODO: refactor

print('n_fft', n_fft)
print('hop_length', hop_length)
print('n_fft secs', n_fft * (1 / rate))
print('hop_length secs', hop_length * (1 / rate))

# method 1
x = librosa.core.stft(sig, n_fft=n_fft, hop_length=hop_length)
x = np.abs(x)
x = np.dot(librosa.filters.mel(rate, n_fft), x)
x = np.log(x + 1e-7)  # TODO: add eps?

# normalize
x = (x - x.mean()) / x.std()
# x = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)

# pad
x = np.concatenate([x, np.zeros_like(x)], 1)

# print(x.mean(), x.std())


print(x.shape, x.dtype)
print(x.min(), x.max(), x.std())

plt.hist(x.ravel(), bins=100)
plt.show()

print(x[-10:, -10:])

plt.imshow(x)
plt.show()
