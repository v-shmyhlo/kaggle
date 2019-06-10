import pandas as pd
import os
from tqdm import tqdm
import torch
import torch.utils
import torch.utils.data
import torchvision.transforms as T
import utils
from .model import Model
from .dataset import NUM_CLASSES, ID_TO_CLASS, TestDataset, load_test_data
from .utils import collate_fn
from frees.transform import ToTensor, LoadSignal, TTA

FOLDS = list(range(1, 5 + 1))
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
WORKERS = 0


class Config(object):
    class Model(object):
        type = 'resnet18-maxpool-2d'
        dropout = 0.2
        sample_rate = 44100

    class Aug(object):
        class Crop(object):
            size = 15

        type = 'crop'
        crop = Crop()

    seed = 42
    batch_size = 50
    model = Model()
    aug = Aug()


config = Config()

if config.aug.type == 'pad':
    test_transform = T.Compose([
        LoadSignal(config.model.sample_rate),
        TTA(),
        T.Lambda(lambda xs: torch.stack([ToTensor()(x) for x in xs], 0)),
    ])
elif config.aug.type == 'crop':
    test_transform = T.Compose([
        LoadSignal(config.model.sample_rate),
        TTA(),
        T.Lambda(lambda xs: torch.stack([ToTensor()(x) for x in xs], 0)),
    ])
else:
    raise AssertionError('invalid aug {}'.format(config.aug.type))


def worker_init_fn(_):
    utils.seed_python(torch.initial_seed() % 2**32)


def rankdata(input, axis=None):
    return input
    # return input.argsort(axis).argsort(axis).float()


def build_submission(model_paths, folds, test_data):
    with torch.no_grad():
        predictions = 0.

        for model_path in model_paths:
            for fold in folds:
                fold_predictions, fold_ids = predict_on_test_using_fold(model_path, fold, test_data)
                predictions = predictions + fold_predictions
                ids = fold_ids

        size = len(model_paths) * len(folds)
        predictions = predictions / size

        return predictions, ids


def predict_on_test_using_fold(model_path, fold, test_data):
    test_dataset = TestDataset(test_data, transform=test_transform)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size // 3,
        num_workers=WORKERS,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn)

    model = Model(config.model, NUM_CLASSES)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(model_path, 'model_{}.pth'.format(fold))))

    model.eval()
    with torch.no_grad():
        fold_predictions = []
        fold_ids = []
        for sigs, ids in tqdm(test_data_loader, desc='fold {} inference'.format(fold)):
            b, n, w = sigs.size()
            sigs = sigs.view(b * n, w)
            sigs = sigs.to(DEVICE)
            logits, _, _ = model(sigs)
            logits = logits.view(b, n, NUM_CLASSES)
            logits = rankdata(logits, -1).mean(1)

            fold_predictions.append(logits)
            fold_ids.extend(ids)

        fold_predictions = torch.cat(fold_predictions, 0)

    return fold_predictions, fold_ids


def main(model_paths, dataset_path, submission_path):
    utils.seed_python(config.seed)
    utils.seed_torch(config.seed)

    test_data = load_test_data(dataset_path, 'test')
    predictions, ids = build_submission(model_paths, FOLDS, test_data)
    predictions = predictions.cpu()
    submission = {
        'fname': ids,
        **{ID_TO_CLASS[i]: predictions[:, i] for i in range(NUM_CLASSES)}
    }
    submission = pd.DataFrame(submission)
    submission.to_csv(submission_path, index=False)
