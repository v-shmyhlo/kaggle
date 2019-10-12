import numpy as np
import torch


def rle_encode(image):
    dots = np.where(image.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2

    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))

        run_lengths[-1] += 1
        prev = b

    return run_lengths


def rle_decode(rle, size):
    starts, lengths = [np.asarray(x) for x in (rle[0:][::2], rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    image = np.zeros(size[0] * size[1], dtype=np.bool)

    for lo, hi in zip(starts, ends):
        image[lo:hi] = 1

    return image.reshape((size[1], size[0])).T


def mask_to_image(mask, num_classes):
    colors = np.random.RandomState(42).uniform(0.5, 1., size=(num_classes, 3))
    colors[0] = 0.
    colors = torch.tensor(colors, dtype=mask.dtype, device=mask.device)
   
    mask = mask.unsqueeze(2)
    colors = colors.view(1, *colors.size(), 1, 1)

    image = mask * colors
    image = image.sum(1)

    return image
