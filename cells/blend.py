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

# TODO: check all sharpen usage

FOLDS = list(range(1, 3 + 1))
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, action='append', required=True)
parser.add_argument('--experiment-path', type=str, default='./tf_log/cells')
parser.add_argument('--dataset-path', type=str, required=True)
parser.add_argument('--alpha', type=float, required=True)
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


def sharpen(prob, temp):
    assert prob.dim() == 2

    prob = prob**temp
    prob = prob / prob.sum(1, keepdim=True)

    return prob


# TODO: use pool
def find_temp_global(probs, target, exps):
    temps = np.logspace(np.log(1e-4), np.log(1.), 50, base=np.e)
    metrics = []
    for temp in tqdm(temps, desc='temp search'):
        preds = assign_classes(probs=sharpen(probs, temp).data.cpu().numpy(), exps=exps)
        preds = torch.tensor(preds).to(probs.device)
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


def assign_classes(probs, exps, return_cost=False):
    # TODO: refactor numpy/torch usage

    exps = np.array(exps)
    costs = {}
    classes = np.zeros(probs.shape[0], dtype=np.int64)
    for exp in np.unique(exps):
        subset = exps == exp
        preds = probs[subset]
        cost, c, _ = lap.lapjv(1 - preds, extend_cost=True)

        costs[exp] = cost
        classes[subset] = c

    if return_cost:
        return classes, costs
    else:
        return classes


def build_submission(inputs, test_data, temp):
    with torch.no_grad():
        probs, exps, plates, ids = load_data(inputs, 'test.pth')

        classes = assign_classes(probs=sharpen(probs, temp).data.cpu().numpy(), exps=exps)
        probs = refine_probs(probs, classes, exps=exps, plates=plates)
        classes, costs = assign_classes(probs=sharpen(probs, temp).data.cpu().numpy(), exps=exps, return_cost=True)

        tmp = test_data.copy()
        tmp['sirna'] = classes
        tmp.to_csv(os.path.join(args.experiment_path, 'test.csv'), index=False)

        submission = pd.DataFrame({'id_code': ids, 'sirna': classes})
        submission.to_csv(os.path.join(args.experiment_path, 'submission.csv'), index=False)
        submission.to_csv('./submission.csv', index=False)

        submission = pd.DataFrame({'experiment': sorted(costs), 'cost': [costs[k] for k in sorted(costs)]})
        submission.to_csv(os.path.join(args.experiment_path, 'cost.csv'), index=False)
        submission.to_csv('./cost.csv', index=False)


def refine_probs(probs, classes, exps, plates):
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
                probs[subset, ..., i] = 0.

    probs /= probs.sum(-1, keepdim=True)
    assert torch.allclose(probs.sum(-1, keepdim=True), torch.ones_like(probs))

    return probs


def find_temp_for_folds(inputs):
    with torch.no_grad():
        labels, probs, exps, plates, ids = load_data(inputs, 'oof.pth')

        temp, _, _ = find_temp_global(probs=probs, target=labels, exps=exps)
        classes = assign_classes(probs=sharpen(probs, temp).data.cpu().numpy(), exps=exps)
        probs = refine_probs(probs, classes, exps=exps, plates=plates)
        classes = assign_classes(probs=sharpen(probs, temp).data.cpu().numpy(), exps=exps)

        metric = compute_metric(input=torch.tensor(classes).to(probs.device), target=labels, exps=exps)
        print('metric: {:.4f}, temp: {:.4f}'.format(metric['accuracy@1'].mean().data.cpu().numpy(), temp))

        submission = pd.DataFrame({'id_code': ids, 'sirna': classes})
        submission.to_csv(os.path.join(args.experiment_path, 'eval.csv'), index=False)
        submission.to_csv('./eval.csv', index=False)

        return temp


def load_data(inputs, name):
    w = [args.alpha, 1. - args.alpha]
    assert sum(w) == 1.
    assert len(inputs) == len(w)

    if name == 'oof.pth':
        probs = 0.

        for i, input in enumerate(inputs):
            labels, model_probs, exps, plates, ids = torch.load(os.path.join(input, name), map_location=DEVICE)
            assert torch.allclose(model_probs.sum(-1, keepdim=True), torch.ones_like(model_probs))
            probs += model_probs * w[i]

        return labels, probs, exps, plates, ids
    elif name == 'test.pth':
        probs = 0.

        for i, input in enumerate(inputs):
            model_probs, exps, plates, ids = torch.load(os.path.join(input, name), map_location=DEVICE)
            assert torch.allclose(model_probs.sum(-1, keepdim=True), torch.ones_like(model_probs))
            probs += model_probs * w[i]

        return probs, exps, plates, ids
    else:
        raise AssertionError('invalid name {}'.format(name))


def main():
    test_data = pd.read_csv(os.path.join(args.dataset_path, 'test.csv'))
    test_data['root'] = os.path.join(args.dataset_path, 'test')

    temp = find_temp_for_folds(args.input)
    gc.collect()
    build_submission(args.input, test_data, temp)


if __name__ == '__main__':
    main()
