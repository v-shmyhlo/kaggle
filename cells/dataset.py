import os

import torch
from PIL import Image

NUM_CLASSES = 1108


class TrainEvalDataset(torch.utils.data.Dataset):
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
            image.extend(load_image(row['root'], row['experiment'], row['plate'], row['well'], s))

        cell_type = row['experiment'].split('-')[0]
        feat = torch.tensor([self.cell_type_to_id[cell_type], row['plate'] - 1])

        input = {
            'image': image,
            'feat': feat,
            'exp': row['experiment'],
            'plate': row['plate'],
            'label': row['sirna'],
            # 'real': torch.tensor(row['real']),  # TODO:
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
            image.extend(load_image(row['root'], row['experiment'], row['plate'], row['well'], s))

        cell_type = row['experiment'].split('-')[0]
        feat = torch.tensor([self.cell_type_to_id[cell_type], row['plate'] - 1])

        input = {
            'image': image,
            'feat': feat,
            'exp': row['experiment'],
            'plate': row['plate'],
            'id': row['id_code']
        }

        if self.transform is not None:
            input = self.transform(input)

        return input


def load_image(root, experiment, plate, well, site):
    image = []

    for c in range(1, 7):
        path = os.path.join(
            root,
            experiment,
            'Plate{}'.format(plate),
            '{}_s{}_w{}.png'.format(well, site, c))

        image.append(Image.open(path))

    return image
