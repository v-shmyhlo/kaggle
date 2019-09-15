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


def cut_mix(images_1, labels_1):
    b, _, h, w = images_1.size()
    perm = np.random.permutation(b)
    images_2, labels_2 = images_1[perm], labels_1[perm]

    lam = np.random.uniform(0, 1)
    r_x = np.random.uniform(0, w)
    r_y = np.random.uniform(0, h)
    r_w = w * np.sqrt(1 - lam)
    r_h = h * np.sqrt(1 - lam)
    x1 = (r_x - r_w / 2).clip(0, w).round().astype(np.int32)
    x2 = (r_x + r_w / 2).clip(0, w).round().astype(np.int32)
    y1 = (r_y - r_h / 2).clip(0, h).round().astype(np.int32)
    y2 = (r_y + r_h / 2).clip(0, h).round().astype(np.int32)

    images_1[:, :, x1:x2, y1:y2] = images_2[:, :, x1:x2, y1:y2]
    images = images_1
    labels = lam * labels_1 + (1 - lam) * labels_2

    return images, labels
