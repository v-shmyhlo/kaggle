import os

import torch
from PIL import Image

NUM_CLASSES = 1108


class TrainEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        # self.experiment_to_id = {e: i for i, e in enumerate(sorted(data['experiment'].unique()))}
       
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        row = self.data.iloc[item]

        image = []
        for s in [1, 2]:
            image.extend([
                Image.open(os.path.join(
                    row['root'],
                    row['experiment'],
                    'Plate{}'.format(row['plate']),
                    '{}_s{}_w{}.png'.format(row['well'], s, c)))
                for c in range(1, 7)])

        input = {
            'image': image,
            'label': row['sirna'],
            # 'exp': self.experiment_to_id[row['experiment']],
            'id': row['id_code']
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
        row = self.data.iloc[item]

        image = []
        for s in [1, 2]:
            image.extend([
                Image.open(os.path.join(
                    row['root'],
                    row['experiment'],
                    'Plate{}'.format(row['plate']),
                    '{}_s{}_w{}.png'.format(row['well'], s, c)))
                for c in range(1, 7)])

        input = {
            'image': image,
            'id': row['id_code']
        }

        if self.transform is not None:
            input = self.transform(input)

        return input
