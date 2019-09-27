import argparse
import gc
import os

import editdistance
import lap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributions
import torch.utils
import torch.utils.data
from tqdm import tqdm

from cells.dataset import NUM_CLASSES

FOLDS = list(range(1, 3 + 1))
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, action='append', required=True)
parser.add_argument('--experiment-path', type=str, default='./tf_log/cells')
parser.add_argument('--dataset-path', type=str, required=True)
args = parser.parse_args()
os.makedirs(args.experiment_path, exist_ok=True)

groups = pd.read_csv(os.path.join(args.dataset_path, 'train.csv'))
groups = groups.groupby(['experiment', 'plate'])['sirna'].apply(sorted).apply(tuple)
groups = groups[groups.apply(len) == NUM_CLASSES // 4].unique()
assert len(groups) == 4
x = np.zeros((4, 4), dtype=np.int32)
for i in range(4):
    for j in range(4):
        x[i, j] = len(set(groups[i]).intersection(set(groups[j])))
print(x)


def to_prob(input, temp):
    if input.dim() == 2:
        # (B, C)
        input = (input * temp).softmax(1)
    elif input.dim() == 3:
        # (B, N, C)
        input = (input * temp).softmax(2).mean(1)
    else:
        raise AssertionError('invalid input shape: {}'.format(input.size()))

    return input


# TODO: use pool
def find_temp_global(input, target, exps):
    temps = np.logspace(np.log(1e-4), np.log(1.), 100, base=np.e)
    metrics = []
    for temp in tqdm(temps, desc='temp search'):
        preds = assign_classes(probs=to_prob(input, temp).data.cpu().numpy(), exps=exps)
        preds = torch.tensor(preds).to(input.device)
        metric = compute_metric(input=preds, target=target, exps=exps)
        metrics.append(metric['accuracy@1'].mean().data.cpu().numpy())

    temp = temps[np.argmax(metrics)]
    metric = metrics[np.argmax(metrics)]
    fig = plt.figure()
    plt.plot(temps, metrics)
    plt.xscale('log')
    plt.axvline(temp)
    plt.title('metric: {:.4f}, temp: {:.4f}'.format(metric.item(), temp))
    plt.savefig('./fig.png')

    return temp, metric.item(), fig


def compute_metric(input, target, exps):
    exps = np.array(exps)
    metric = {
        'accuracy@1': (input == target).float(),
    }

    for exp in np.unique(exps):
        mask = torch.tensor(exp == exps)
        metric['experiment/{}/accuracy@1'.format(exp)] = (input[mask] == target[mask]).float()

    return metric


def assign_classes(probs, exps):
    # TODO: refactor numpy/torch usage

    exps = np.array(exps)
    classes = np.zeros(probs.shape[0], dtype=np.int64)
    for exp in np.unique(exps):
        subset = exps == exp
        preds = probs[subset]
        _, c, _ = lap.lapjv(1 - preds, extend_cost=True)
        classes[subset] = c

    return classes


def build_submission(inputs, folds, test_data, temp):
    with torch.no_grad():
        probs = 0.

        for fold in folds:
            fold_logits, fold_exps, fold_ids = load_data(inputs, 'test_{}.pth'.format(fold))
            fold_probs = to_prob(fold_logits, temp)

            probs = probs + fold_probs
            exps = fold_exps
            ids = fold_ids

        probs = probs / len(folds)
        torch.save(probs, os.path.join(args.experiment_path, 'test.pth'))
        probs = probs.data.cpu().numpy()
        assert len(probs) == len(exps) == len(ids)
        classes = assign_classes(probs=probs, exps=exps)
        probs = refine_scores(
            probs, classes, exps=exps, plates=test_data['plate'].values, value=0.)
        classes = assign_classes(probs=probs, exps=exps)

        tmp = test_data.copy()
        tmp['sirna'] = classes
        tmp.to_csv(os.path.join(args.experiment_path, 'test.csv'), index=False)

        submission = pd.DataFrame({'id_code': ids, 'sirna': classes})
        submission.to_csv(os.path.join(args.experiment_path, 'submission.csv'), index=False)
        submission.to_csv('./submission.csv', index=False)


def refine_scores(logits, classes, exps, plates, value):
    exps = np.array(exps)
    plates = np.array(plates)

    for exp in np.unique(exps):
        exp_subset = exps == exp

        for plate in np.unique(plates):
            plate_subset = plates == plate
            subset = exp_subset & plate_subset

            c = classes[subset]
            c = tuple(sorted(c))

            d = np.array([editdistance.eval(c, g) for g in groups])
            g = groups[np.argmin(d)]

            ignored = set(range(NUM_CLASSES)) - set(g)
            subset = torch.tensor(subset)
            for i in ignored:
                # FIXME:
                logits[subset, ..., i] = value

    return logits


def find_temp_for_folds(inputs):
    with torch.no_grad():
        labels, logits, exps, ids = load_data(inputs, 'oof.pth')
        temp, metric, _ = find_temp_global(input=logits, target=labels, exps=exps)
        print('metric: {:.4f}, temp: {:.4f}'.format(metric, temp))

        classes = assign_classes(probs=to_prob(logits, temp).data.cpu().numpy(), exps=exps)
        submission = pd.DataFrame({'id_code': ids, 'sirna': classes})
        submission.to_csv(os.path.join(args.experiment_path, 'eval.csv'), index=False)
        submission.to_csv('./eval.csv', index=False)

        return temp


def main():
    test_data = pd.read_csv(os.path.join(args.dataset_path, 'test.csv'))
    test_data['root'] = os.path.join(args.dataset_path, 'test')

    temp = find_temp_for_folds(args.input)
    gc.collect()
    folds = FOLDS
    build_submission(args.input, folds, test_data, temp)


def load_data(inputs, name):
    if name.startswith('oof'):
        logits = []
        for i, input in enumerate(inputs):
            labels, model_logits, exps, ids = torch.load(os.path.join(input, name), map_location=DEVICE)
            print(model_logits.shape)
            logits.append(model_logits)
        logits = torch.cat(logits, 1)
        print(logits.shape)

        return labels, logits, exps, ids
    elif name.startswith('test'):
        logits = []
        for i, input in enumerate(inputs):
            model_logits, exps, ids = torch.load(os.path.join(input, name), map_location=DEVICE)
            print(model_logits.shape)
            logits.append(model_logits)
        logits = torch.cat(logits, 1)
        print(logits.shape)

        return logits, exps, ids
    else:
        raise AssertionError('invalid name {}'.format(name))


if __name__ == '__main__':
    main()
