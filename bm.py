import os
from multiprocessing import Pool

import numpy as np
from pysndfx import AudioEffectsChain
from tqdm import tqdm


def worker(_):
    sig = np.random.standard_normal(44100 * 8)

    effect = AudioEffectsChain()
    effect = effect.pitch(np.random.uniform(-300, 300))
    effect = effect.tempo(np.random.uniform(0.8, 1.2))
    effect = effect.reverb(np.random.uniform(0, 100))

    return effect(sig)


with Pool(os.cpu_count()) as pool:
    tasks = tqdm(pool.imap(worker, range(100000)))

    for _ in tasks:
        pass
