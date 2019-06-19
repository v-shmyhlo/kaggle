import os

import pycocotools.coco as pycoco
import torch
import torch.utils.data
from PIL import Image

NUM_CLASSES = 80


# TODO: refactor
class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, train, transform=None):
        if train:
            ann_path = os.path.join(path, 'annotations/instances_train2017.json')
            self.path = os.path.join(path, 'train2017')
        else:
            ann_path = os.path.join(path, 'annotations/instances_val2017.json')
            self.path = os.path.join(path, 'val2017')

        self.transform = transform
        self.coco = pycoco.COCO(ann_path)
        self.cat_to_id = {c: i for i, c in enumerate(self.coco.getCatIds())}
        assert len(self.cat_to_id) == NUM_CLASSES
        self.data = self.coco.loadImgs(ids=self.coco.getImgIds())
        self.data = [item for item in self.data
                     if len(self.coco.loadAnns(ids=self.coco.getAnnIds(imgIds=item['id'], iscrowd=False))) > 0]

    def __len__(self):
        return len(self.data)

    # TODO: check
    def __getitem__(self, item):
        item = self.data[item]

        image = Image.open(os.path.join(self.path, item['file_name']))
        if image.mode == 'L':
            image = image.convert('RGB')

        annotation_ids = self.coco.getAnnIds(imgIds=item['id'], iscrowd=False)
        annotations = self.coco.loadAnns(ids=annotation_ids)

        boxes = []
        class_ids = []
        for a in annotations:
            l, t, w, h = a['bbox']
            y = t + h / 2
            x = l + w / 2

            boxes.append([y, x, h, w])
            class_ids.append(self.cat_to_id[a['category_id']])

        class_ids = torch.tensor(class_ids).view(-1).long()
        boxes = torch.tensor(boxes).view(-1, 4).float()

        input = image, (class_ids, boxes)

        if self.transform is not None:
            input = self.transform(input)

        return input
