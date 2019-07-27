import numpy as np


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
       
    return image.reshape(size).T
