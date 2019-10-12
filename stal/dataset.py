import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from stal.utils import rle_decode

NUM_CLASSES = 5


class TrainEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data.iloc[item]

        image = Image.open(os.path.join(sample['root'], sample['id']))
        image = np.array(image)
        image = (image / 255).astype(np.float32)

        mask = np.zeros(image.shape[:2], dtype=np.int32)
        for i, rle in enumerate(sample['rles'], 1):
            m = rle_decode(rle, image.shape[:2])
            assert m.dtype == np.bool
            assert np.all(mask[m] == 0)
            mask[m] = i
        mask = np.eye(NUM_CLASSES, dtype=np.float32)[mask]

        assert np.allclose(mask.sum(2), np.ones(mask.shape[:2]))
        assert image.shape[:2] == mask.shape[:2] == (256, 1600)

        input = {
            'image': image,
            'mask': mask,
            'id': sample['id'],
        }

        if self.transform is not None:
            input = self.transform(input)

        return input


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data.iloc[item]

        image = Image.open(os.path.join(sample['root'], sample['id']))
        assert image.size == (1600, 256)

        input = {
            'image': image,
            'id': sample['id'],
        }

        if self.transform is not None:
            input = self.transform(input)

        return input


def build_data(data):
    id_root_to_sample = defaultdict(lambda: [[]] * 4)
    for _, row in tqdm(data.iterrows(), desc='building data'):
        image_id, class_id = row['ImageId_ClassId'].split('_')
        id_root_to_sample[(image_id, row['root'])][int(class_id) - 1] = [int(x) for x in row['EncodedPixels'].split()]

    data = {
        'id': [],
        'rles': [],
        'root': [],
    }
    for id, root in sorted(id_root_to_sample):
        data['id'].append(id)
        data['rles'].append(id_root_to_sample[(id, root)])
        data['root'].append(root)

    data = pd.DataFrame(data)

    return data
