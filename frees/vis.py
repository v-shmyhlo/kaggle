import numpy as np
import matplotlib.pyplot as plt
import soundfile
import scipy.signal
import librosa

sig, rate = soundfile.read('./frees/sample.wav', dtype=np.float32)
sig = sig[:, 0]

n_fft = round(0.025 * rate)  # TODO: refactor
hop_length = round(0.01 * rate)  # TODO: refactor

print(n_fft, hop_length)
print(n_fft * (1 / rate), hop_length * (1 / rate))

# x = librosa.feature.melspectrogram(sig, n_fft=n_fft, hop_length=hop_length)
x = librosa.core.stft(sig, n_fft=n_fft, hop_length=hop_length)
x = np.abs(x)
x = np.dot(librosa.filters.mel(rate, n_fft), x)
x = np.log(x)
# x = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
x = (x - x.mean()) / x.std()

# librosa.power_to_db
# librosa.feature.melspectrogram

print(x.shape, x.dtype)
print(x.min(), x.max(), x.std())

plt.hist(x.ravel(), bins=100)
plt.show()

plt.imshow(x[:, :500])
plt.show()
