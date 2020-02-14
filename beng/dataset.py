import glob
import os

import numpy as np
import pandas as pd
import torch.utils.data
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from all_the_tools.torch.utils import one_hot
from beng.transforms import invert

IMAGE_SIZE = 137, 236

CLASS_META = pd.DataFrame({
    'component': ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic'],
    'num_classes': [168, 7, 11],
    'weight': [2, 1, 1],
}, index=[0, 1, 2])


class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data.iloc[i]

        image = Image.open(item['image_path'])

        if self.transform is not None:
            image = self.transform(image)

        target = torch.tensor([
            item[component]
            for component in CLASS_META['component']
        ], dtype=torch.long)
        target = encode_target(target)

        return image, target


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, data, size, transform=None):
        self.data = data
        self.size = size
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        i = np.random.choice(len(self))
        item = self.data.iloc[i]

        font = ImageFont.truetype(np.random.choice(glob.glob('./beng/fonts/*.ttf')), 120)
        image = char_to_image(item['grapheme'], font)

        if self.transform is not None:
            image = self.transform(image)

        target = torch.tensor([
            item[component]
            for component in CLASS_META['component']
        ], dtype=torch.long)
        target = encode_target(target)

        return image, target


def load_labeled_data(metadata_path, parquet_paths, cache_path):
    data = pd.read_csv(metadata_path)
    data = data.set_index('image_id')
    data['image_path'] = data.index.map(lambda id: os.path.join(cache_path, '{}.png'.format(id)))

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

        for parquet_path in tqdm(parquet_paths, desc='loading parquet files'):
            parquet = pd.read_parquet(parquet_path)
            parquet = parquet.set_index('image_id')
            images = parquet.values.reshape(len(parquet), *IMAGE_SIZE)

            for id, image in tqdm(
                    zip(parquet.index, images), total=len(parquet), desc='loading {}'.format(parquet_path)):
                Image.fromarray(image).save(os.path.join(cache_path, '{}.png'.format(id)))

    return data


def split_target(target, dim=None):
    return torch.split(target, CLASS_META['num_classes'].values.tolist(), dim=dim)


def encode_target(target):
    target = target.unbind(-1)
    target = [one_hot(x, num_classes) for x, num_classes in zip(target, CLASS_META['num_classes'])]
    target = torch.cat(target, -1)

    return target


def decode_target(target):
    target = split_target(target, -1)
    target = [x.argmax(-1) for x in target]
    target = torch.stack(target, -1)

    return target


def char_to_image(char, font):
    image = Image.new('L', (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    draw = ImageDraw.Draw(image)
    w, h = draw.textsize(char, font=font)
    draw.text(((IMAGE_SIZE[1] - w) / 2, (IMAGE_SIZE[0] - h) / 3), char, font=font, fill=255)
    image = invert(image)

    return image
