import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

NUM_CLASSES = 1108


class TrainEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.cell_type_to_id = {cell_type: i for i, cell_type in enumerate(['HEPG2', 'HUVEC', 'RPE', 'U2OS'])}

        self.tmp = {}
        controls = pd.concat([
            pd.read_csv('../../data/cells/train_controls.csv'),
            pd.read_csv('../../data/cells/test_controls.csv'),
        ])
        controls = controls[controls['sirna'] == 1138]
        for _, row in controls.iterrows():
            k = '{}_{}'.format(row['experiment'], row['plate'])
            if k not in self.tmp:
                self.tmp[k] = []
            self.tmp[k].append(row['well'])

        # print(len(self.tmp))
        # import numpy as np
        # print(np.unique([len(self.tmp[k]) for k in self.tmp]))

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

        ref_well = self.tmp['{}_{}'.format(row['experiment'], row['plate'])]  # TODO: 0 well
        ref_well = np.random.choice(ref_well)
        ref = []
        for s in [1, 2]:
            ref.extend([
                Image.open(os.path.join(
                    row['root'],
                    row['experiment'],
                    'Plate{}'.format(row['plate']),
                    '{}_s{}_w{}.png'.format(ref_well, s, c)))
                for c in range(1, 7)])

        cell_type = row['experiment'].split('-')[0]
        feat = self.cell_type_to_id[cell_type]

        input = {
            'image': image,
            'ref': ref,
            'feat': feat,
            'label': row['sirna'],
            'id': row['id_code'],
        }

        if self.transform is not None:
            input = self.transform(input)

        return input


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.cell_type_to_id = {cell_type: i for i, cell_type in enumerate(['HEPG2', 'HUVEC', 'RPE', 'U2OS'])}

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

        cell_type = row['experiment'].split('-')[0]
        feat = self.cell_type_to_id[cell_type]

        input = {
            'image': image,
            'feat': feat,
            'id': row['id_code']
        }

        if self.transform is not None:
            input = self.transform(input)

        return input
