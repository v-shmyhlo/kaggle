import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from stal.utils import rle_decode

NUM_CLASSES = 5


class TrainEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = build_data(data)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        id, rles, root = self.data[item]

        image = Image.open(os.path.join(root, id))
        mask = np.zeros((image.size[1], image.size[0]), dtype=np.int32)
        for i, rle in enumerate(rles, 1):
            m = rle_decode(rle, image.size)
            assert m.dtype == np.bool
            assert np.all(mask[m] == 0)
            mask[m] = i
        mask = Image.fromarray(mask)
        assert image.size == mask.size

        input = {
            'image': image,
            'mask': mask,
            'id': id,
        }

        if self.transform is not None:
            input = self.transform(input)

        return input


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.plate_to_stats = torch.load('./cells/plate_stats.pth')
        self.cell_type_to_id = {cell_type: i for i, cell_type in enumerate(['HEPG2', 'HUVEC', 'RPE', 'U2OS'])}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        id, rles = self.data.iloc[item]

        image = []
        for s in [1, 2]:
            image.extend(load_image(row['root'], row['experiment'], row['plate'], row['well'], s))

        ref_stats = self.plate_to_stats['{}_{}'.format(row['experiment'], row['plate'])]
        cell_type = row['experiment'].split('-')[0]
        feat = torch.tensor([self.cell_type_to_id[cell_type], row['plate'] - 1])

        input = {
            'image': image,
            'feat': feat,
            'exp': row['experiment'],
            'ref_stats': ref_stats,
            'id': row['id_code']
        }

        if self.transform is not None:
            input = self.transform(input)

        return input


def build_data(data):
    result = defaultdict(lambda: [[]] * 4)
    for _, row in tqdm(data.iterrows(), desc='building data'):
        image_id, class_id = row['ImageId_ClassId'].split('_')
        result[(image_id, row['root'])][int(class_id) - 1] = [int(x) for x in row['EncodedPixels'].split()]

    result = [(id, result[(id, root)], root) for id, root in result]

    return result
