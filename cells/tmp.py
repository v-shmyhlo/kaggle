import lap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


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


def compute_metric(input, target):
    metric = {
        'accuracy@1': (input == target).float(),
    }

    return metric


def find_temp_global(input, target, exps):
    temps = np.logspace(np.log(0.001), np.log(1.), 30, base=np.e)
    metrics = []
    for temp in tqdm(temps, desc='temp search'):
        fold_preds = assign_classes(probs=(input * temp).softmax(1).data.cpu().numpy(), exps=exps)
        fold_preds = torch.tensor(fold_preds).to(input.device)
        metric = compute_metric(input=fold_preds, target=target)
        metrics.append(metric['accuracy@1'].mean().data.cpu().numpy())

    temp = temps[np.argmax(metrics)]
    metric = metrics[np.argmax(metrics)]
    fig = plt.figure()
    plt.plot(temps, metrics)
    plt.xscale('log')
    plt.axvline(temp)
    plt.title('metric: {:.4f}, temp: {:.4f}'.format(metric.item(), temp))

    return temp, metric.item(), fig


with torch.no_grad():
    folds = [1, 2, 3]
    labels, logits, exps, _ = torch.load('./oof.pth')
    temp, metric, _ = find_temp_global(input=logits, target=labels, exps=exps)
    print('metric: {:.4f}, temp: {:.4f}'.format(metric, temp))

    probs = 0.
    for fold in folds:
        fold_logits, fold_exps, fold_ids = torch.load('./test_{}.pth'.format(fold))
        fold_probs = (fold_logits * temp).softmax(2).mean(1)

        probs = probs + fold_probs
        exps = fold_exps
        ids = fold_ids

    probs = probs / len(folds)
    probs = probs.data.cpu().numpy()
    assert len(probs) == len(exps) == len(ids)
    classes = assign_classes(probs=probs, exps=exps)

    submission = pd.DataFrame({'id_code': ids, 'sirna': classes})
    submission.to_csv('./submission.csv', index=False)
