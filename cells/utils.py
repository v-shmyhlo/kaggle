import numpy as np
import torch

COLORS = torch.tensor([
    [19, 0, 249],
    [42, 255, 31],
    [255, 0, 25],
    [45, 255, 252],
    [250, 0, 253],
    [254, 255, 40],
], dtype=torch.float) / 255

RANGES = torch.tensor([
    [0, 51],
    [0, 107],
    [0, 64],
    [0, 191],
    [0, 89],
    [0, 191],
], dtype=torch.float) / 255


def images_to_rgb(input):
    colors, ranges = COLORS.to(input.device), RANGES.to(input.device)

    min, max = input.min(), input.max()
    input = (input - min) / (max - min)

    colors = colors.reshape((1, 6, 3, 1, 1))
    input = input.unsqueeze(2)
    input = input * colors
    input = input.mean(1)

    return input


def mixup(images_1, labels_1):
    b = images_1.size(0)
    perm = np.random.permutation(b)
    images_2, labels_2 = images_1[perm], labels_1[perm]

    lam = np.random.beta(0.75, 0.75)
    lam = np.maximum(lam, 1 - lam)
    images = lam * images_1 + (1 - lam) * images_2
    labels = lam * labels_1 + (1 - lam) * labels_2

    return images, labels
