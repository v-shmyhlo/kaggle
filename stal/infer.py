import gc
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.distributions
import torch.distributions
import torch.utils
import torch.utils
import torch.utils.data
import torch.utils.data
import torchvision.transforms as T
from tqdm import tqdm

from config import Config
from stal.dataset import NUM_CLASSES, TestDataset
from stal.model_cls import Model
from stal.transforms import ApplyTo, Extract
from stal.utils import rle_encode

FOLDS = list(range(1, 5 + 1))
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_transform = T.Compose([
    ApplyTo(
        ['image'],
        T.Lambda(lambda x: torch.stack([T.ToTensor()(x)], 0))),
    Extract(['image', 'id']),
])


def update_transforms(p):
    assert 0. <= p <= 1.


def worker_init_fn(_):
    seed_python(torch.initial_seed() % 2**32)


def one_hot(input):
    input = torch.eye(NUM_CLASSES).to(input.device)[input]
    input = input.permute((0, 3, 1, 2))

    return input


def seed_python(seed):
    # os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    # tf.set_random_seed(seed)
    np.random.seed(seed)


def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_submission(folds, test_data, temp, experiment_path, config, workers):
    with torch.no_grad():
        for fold in folds:
            fold_rles, fold_ids = predict_on_test_using_fold(
                fold, test_data, experiment_path=experiment_path, config=config, workers=workers)

            rles = fold_rles
            ids = fold_ids

        submission_rles = []
        submission_ids = []
        for rle4, id in zip(rles, ids):
            submission_rles.extend([' '.join(map(str, rle)) for rle in rle4])
            submission_ids.extend(['{}_{}'.format(id, n) for n in range(1, 5)])
        assert len(submission_rles) == len(submission_ids)

        submission = pd.DataFrame({'ImageId_ClassId': submission_ids, 'EncodedPixels': submission_rles})
        submission.to_csv('./submission.csv', index=False)


def predict_on_test_using_fold(fold, test_data, experiment_path, config, workers):
    test_dataset = TestDataset(test_data, transform=test_transform)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size // 2,
        num_workers=workers,
        worker_init_fn=worker_init_fn)

    model = Model(config.model, NUM_CLASSES, pretrained=False)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(experiment_path, 'model_{}.pth'.format(fold))))

    model.eval()
    with torch.no_grad():
        fold_rles = []
        fold_ids = []

        for images, ids in tqdm(test_data_loader, desc='fold {} inference'.format(fold)):
            images = images.to(DEVICE)

            b, n, c, h, w = images.size()
            images = images.view(b * n, c, h, w)
            class_logits, mask_logits = model(images)
            class_logits = class_logits.view(b, n, NUM_CLASSES)
            mask_logits = mask_logits.view(b, n, NUM_CLASSES, h, w)

            n_dim = 1
            c_dim = 2

            class_probs = class_logits.sigmoid().mean(n_dim)
            mask_probs = mask_logits.softmax(c_dim).mean(n_dim)

            class_probs = class_probs[:, 1:]
            mask_probs = one_hot(mask_probs.argmax(1))[:, 1:]

            class_probs = (class_probs > 0.5).float().view(class_probs.size(0), class_probs.size(1), 1, 1)
            mask_probs = mask_probs * class_probs

            rles = [
                [rle_encode(c) for c in mask]
                for mask in mask_probs.data.cpu().numpy()
            ]

            fold_rles.extend(rles)
            fold_ids.extend(ids)

    return fold_rles, fold_ids


def main(folds, experiment_path, dataset_path, workers):
    config = Config.from_json(os.path.join(experiment_path, 'config.yaml'))

    seed_python(config.seed)
    seed_torch(config.seed)

    test_ids = []
    for id in os.listdir(os.path.join(dataset_path, 'test_images')):
        test_ids.extend(['{}_{}'.format(id, n) for n in range(1, 5)])
    test_data = pd.DataFrame({'ImageId_ClassId': test_ids, 'EncodedPixels': ['' for _ in test_ids]})
    test_data['root'] = os.path.join(dataset_path, 'test_images')

    update_transforms(1.)  # FIXME:
    # temp = find_temp_for_folds(folds, train_eval_data)
    temp = None
    gc.collect()

    build_submission(folds, test_data, temp, experiment_path=experiment_path, config=config, workers=workers)


if __name__ == '__main__':
    main()
