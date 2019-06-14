# import pandas as pd lr
from torch_scatter import scatter_mean
from torch_geometric.data import DataLoader
import click

#
# filename = './data/mol/structures/dsgdb9nsd_133882.xyz'
#
# from ase import Atoms
import pandas as pd
# import ase.visualize
# # from ase.io import read
#

# # graphs = pd.read_csv('./data/mol/sample_submission.csv')

# TODO: bidirectional edges
# TODO: ohem


import os
import gc
from multiprocessing import Pool
from tensorboardX import SummaryWriter
from tqdm import tqdm
import lr_scheduler_wrapper
from config import Config
import torch
from sklearn.model_selection import KFold
import numpy as np
import ase.io
import ase.neighborlist
import ase.utils
import torch.nn as nn
from lr_scheduler import OneCycleScheduler
import torch.utils.data
import utils
from functools import partial
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MetaLayer

FOLDS = list(range(1, 5 + 1))
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# torch.cuda.set_device(DEVICE)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        id = self.data[item]
        path = './data/mol/graphs/{}.pth'.format(id)

        return torch.load(path)


# class ReLU(nn.SELU):
#     pass

class ReLU(nn.PReLU):
    def __init__(self, inplace):
        super().__init__()


class Layer(nn.Module):
    class EdgeModel(nn.Module):
        def __init__(self, node_features, edge_features, global_features):
            super().__init__()

            self.layer1 = nn.Sequential(
                nn.Linear(node_features[0] + edge_features[0] + global_features[0], edge_features[1]),
                ReLU(inplace=True))

        def forward(self, src, dst, edge_attr, u, batch):
            edge_attr = torch.cat([(src + dst) / 2, edge_attr, u[batch]], 1)
            edge_attr = self.layer1(edge_attr)

            return edge_attr

    class NodeModel(nn.Module):
        def __init__(self, node_features, edge_features, global_features):
            super().__init__()

            self.layer1 = nn.Sequential(
                nn.Linear(node_features[0] + edge_features[1], node_features[1]),
                ReLU(inplace=True))

            self.layer2 = nn.Sequential(
                nn.Linear(node_features[1] + global_features[0], node_features[1]),
                ReLU(inplace=True))

        def forward(self, x, edge_index, edge_attr, u, batch):
            row, col = edge_index
            dim_size, _ = x.size()
            x = torch.cat([x[col], edge_attr], dim=1)
            x = self.layer1(x)
            x = scatter_mean(x, row, dim=0, dim_size=dim_size)
            x = torch.cat([x, u[batch]], dim=1)
            x = self.layer2(x)

            return x

    class GlobalModel(nn.Module):
        def __init__(self, node_features, edge_features, global_features):
            super().__init__()

            self.layer1 = nn.Sequential(
                nn.Linear(global_features[0] + node_features[1], global_features[1]),
                ReLU(inplace=True))

        def forward(self, x, edge_index, edge_attr, u, batch):
            u = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
            u = self.layer1(u)

            return u

    def __init__(self, node_features, edge_features, global_features):
        super().__init__()

        if node_features[1] is None:
            node_model = None
        else:
            node_model = self.NodeModel(
                node_features=node_features, edge_features=edge_features, global_features=global_features)

        if edge_features[1] is None:
            edge_model = None
        else:
            edge_model = self.EdgeModel(
                node_features=node_features, edge_features=edge_features, global_features=global_features)

        if global_features[1] is None:
            global_model = None
        else:
            global_model = self.GlobalModel(
                node_features=node_features, edge_features=edge_features, global_features=global_features)

        self.op = MetaLayer(
            edge_model=edge_model,
            node_model=node_model,
            global_model=global_model)

    def forward(self, x, edge_index, edge_attr, u, batch):
        return self.op(x, edge_index, edge_attr, u, batch)


# TODO: embed size
class Model(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.nodes = nn.Embedding(5, 8)
        self.edges = nn.Embedding(8, 8)

        self.layers = nn.ModuleList([
            Layer(
                node_features=(8 + 3, model.size),
                edge_features=(8 + 4, model.size),
                global_features=(14, model.size)),
            *[Layer(
                node_features=(model.size, model.size),
                edge_features=(model.size, model.size),
                global_features=(model.size, model.size))
                for _ in range(8)],
            Layer(
                node_features=(model.size, None),
                edge_features=(model.size, model.size),
                global_features=(model.size, None))
        ])

        self.output = nn.Linear(model.size, 1)

    def forward(self, batch):
        x, edge_index, edge_attr, u, batch = batch.x, batch.edge_index, batch.edge_attr, batch.u, batch.batch

        x = torch.cat([
            self.nodes(x[:, 0].long()),
            x[:, 1:]
        ], 1)

        edge_attr = torch.cat([
            self.edges(edge_attr[:, 0].long()),
            edge_attr[:, 1:]
        ], 1)

        for l in self.layers:
            x, edge_attr, u = l(x, edge_index, edge_attr, u, batch)

        edge_attr = self.output(edge_attr)

        edge_attr = edge_attr.squeeze(1)

        return edge_attr

        # def forward(self, data):
    # 
    # 
    #     x = self.conv1(x, edge_index)
    #     x = F.relu(x)
    #     x = self.conv2(x, edge_index)
    #     x = F.relu(x)
    #     x = self.conv3(x, edge_index)
    # 
    #     x = x.squeeze(1)
    # 
    #     return x


def worker_init_fn(_):
    utils.seed_python(torch.initial_seed() % 2**32)


# TODO: should use top momentum to pick best lr?
def build_optimizer(optimizer, parameters, lr, beta, weight_decay):
    if optimizer == 'adam':
        return torch.optim.Adam(parameters, lr, betas=(beta, 0.999), weight_decay=weight_decay)
    elif optimizer == 'momentum':
        return torch.optim.SGD(parameters, lr, momentum=beta, weight_decay=weight_decay, nesterov=True)
    elif optimizer == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr, momentum=beta, weight_decay=weight_decay)
    else:
        raise AssertionError('invalid OPT {}'.format(optimizer))


def indices_for_fold(fold, dataset_size, seed):
    kfold = KFold(len(FOLDS), shuffle=True, random_state=seed)
    splits = list(kfold.split(np.zeros(dataset_size)))
    train_indices, eval_indices = splits[fold - 1]
    assert len(train_indices) + len(eval_indices) == dataset_size

    return train_indices, eval_indices


def train_fold(fold, train_eval_data, args, config):
    train_indices, eval_indices = indices_for_fold(fold, len(train_eval_data), seed=config.seed)
    train_dataset = Dataset([train_eval_data[i] for i in train_indices])
    eval_dataset = Dataset([train_eval_data[i] for i in eval_indices])

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=args['workers'],
        worker_init_fn=worker_init_fn)

    eval_data_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        num_workers=args['workers'],
        worker_init_fn=worker_init_fn)

    model = Model(config.model)
    model = model.to(DEVICE)
    lr = 1e-3
    optimizer = build_optimizer(
        config.opt.type, model.parameters(), lr, config.opt.beta, weight_decay=config.opt.weight_decay)

    if config.sched.type == 'onecycle':
        scheduler = lr_scheduler_wrapper.StepWrapper(
            OneCycleScheduler(
                optimizer,
                lr=(lr / 20, lr),
                beta=config.sched.onecycle.beta,
                max_steps=len(train_data_loader) * config.epochs,
                annealing=config.sched.onecycle.anneal))
    elif config.sched.type == 'cyclic':
        scheduler = lr_scheduler_wrapper.StepWrapper(
            torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                0.,
                lr,
                step_size_up=len(train_data_loader),
                step_size_down=len(train_data_loader),
                mode='triangular2',
                cycle_momentum=True,
                base_momentum=config.sched.cyclic.beta[1],
                max_momentum=config.sched.cyclic.beta[0]))
    elif config.sched.type == 'cawr':
        scheduler = lr_scheduler_wrapper.StepWrapper(
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=len(train_data_loader), T_mult=2))
    elif config.sched.type == 'plateau':
        scheduler = lr_scheduler_wrapper.ScoreWrapper(
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=2, verbose=True))
    else:
        raise AssertionError('invalid sched {}'.format(config.sched.type))

    best_score = 0
    for epoch in range(config.epochs):

        train_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data_loader=train_data_loader,
            fold=fold,
            epoch=epoch,
            args=args,
            config=config)
        gc.collect()
        score = eval_epoch(
            model=model,
            data_loader=eval_data_loader,
            fold=fold,
            epoch=epoch,
            args=args,
            config=config)
        gc.collect()

        scheduler.step_epoch()
        scheduler.step_score(score)

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(args['experiment_path'], 'model_{}.pth'.format(fold)))


def compute_loss(input, target, groups):
    groups = groups.long()

    error = torch.abs(input - target)
    error = scatter_mean(error, groups)
    error = torch.clamp(error, 1e-9)
    loss = error.log().mean()

    return loss


def compute_score(input, target, groups):
    groups = groups.long()

    error = torch.abs(input - target)
    error = scatter_mean(error, groups)
    error = torch.clamp(error, 1e-9)
    loss = error.log().mean()

    return loss


def train_epoch(model, optimizer, scheduler, data_loader, fold, epoch, args, config):
    writer = SummaryWriter(os.path.join(args['experiment_path'], 'fold{}'.format(fold), 'train'))

    metrics = {
        'loss': utils.Mean(),
    }

    model.train()
    for batch in tqdm(data_loader, desc='epoch {} train'.format(epoch)):
        batch = batch.to(DEVICE)
        logits = model(batch)

        loss = compute_loss(input=logits, target=batch.y, groups=batch.edge_attr[:, 0])
        metrics['loss'].update(loss.data.cpu().numpy())

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        loss = metrics['loss'].compute_and_reset()

        print('[FOLD {}][EPOCH {}][TRAIN] loss: {:.4f}'.format(fold, epoch, loss))
        writer.add_scalar('loss', loss, global_step=epoch)
        lr, beta = scheduler.get_lr()
        writer.add_scalar('learning_rate', lr, global_step=epoch)
        writer.add_scalar('beta', beta, global_step=epoch)


def eval_epoch(model, data_loader, fold, epoch, args, config):
    writer = SummaryWriter(os.path.join(args['experiment_path'], 'fold{}'.format(fold), 'eval'))

    metrics = {
        'loss': utils.Mean(),
    }

    predictions = []
    targets = []
    groups = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='epoch {} evaluation'.format(epoch)):
            batch = batch.to(DEVICE)
            logits = model(batch)

            targets.append(batch.y)
            predictions.append(logits)
            groups.append(batch.edge_attr[:, 0])

            loss = compute_loss(input=logits, target=batch.y, groups=batch.edge_attr[:, 0])
            metrics['loss'].update(loss.data.cpu().numpy())

        loss = metrics['loss'].compute_and_reset()

        predictions = torch.cat(predictions, 0)
        targets = torch.cat(targets, 0)
        groups = torch.cat(groups, 0)
        score = compute_score(input=predictions, target=targets, groups=groups)

        print('[FOLD {}][EPOCH {}][EVAL] loss: {:.4f}, score: {:.4f}'.format(fold, epoch, loss, score))
        writer.add_scalar('loss', loss, global_step=epoch)
        writer.add_scalar('score', score, global_step=epoch)
        writer.add_histogram('true', batch.y, global_step=epoch)
        writer.add_histogram('pred', logits, global_step=epoch)

        return score


@click.command()
@click.option('--experiment-path', type=click.Path(), required=True)
@click.option('--config-path', type=click.Path(), required=True)
@click.option('--workers', type=click.INT, default=os.cpu_count())
def main(**args):
    config = Config.from_yaml(args['config_path'])

    # graphs = './data/mol/structures'
    # graphs = sorted(os.path.join(graphs, path) for path in os.listdir(graphs))
    # paths = graphs
    # with Pool(os.cpu_count()) as pool:
    #     graphs = pool.map(ase.io.read, tqdm(graphs))
    #
    # symbol_to_index = {}
    # for g in tqdm(graphs):
    #     for a in g:
    #         if a.symbol not in symbol_to_index:
    #             symbol_to_index[a.symbol] = len(symbol_to_index)
    #
    # print(symbol_to_index)
    #
    # with Pool(os.cpu_count()) as pool:
    #     graphs = pool.map(partial(graph_to_data, symbol_to_index=symbol_to_index), tqdm(list(zip(graphs, paths))))

    graphs = sorted(os.path.splitext(path)[0] for path in os.listdir('./data/mol/structures'))
    print(len(graphs))

    train_fold(1, graphs, args=args, config=config)

    # print(graphs[0])
    # bonds =[ ase.neighborlist.neighbor_list('ij', g, ase.utils.natural_cutoffs(g))

    # struct_file = pd.read_csv('./data/mol/structures.csv')
    # molecule = struct_file[struct_file['molecule_name'] == 'dsgdb9nsd_133882']
    # atoms = molecule.iloc[:, 3:].values
    # symbols = molecule.iloc[:, 2].values
    # system = Atoms(positions=atoms, symbols=symbols)
    # print(system)
    #
    # graphs = {}

    # from mol.xyz2mol import xyz2mol, read_xyz_file
    #
    # charged_fragments = True
    # quick = True
    # atomicNumList, charge, xyz_coordinates = read_xyz_file(filename)
    # mol = xyz2mol(atomicNumList, charge, xyz_coordinates, charged_fragments, quick)
    # print(mol)

    # from torch_geometric.datasets import TUDataset
    #
    # from tqdm import tqdm

    # dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

    # from torch_geometric.datasets import Planetoid
    #

    # dataset = Planetoid(root='/tmp/Cora', name='Cora')
    # print(dataset[0])

    # import pandas as pd
    # import os
    # from mol.dataset import Dataset
    #
    # graphs = './data/mol/structures'
    # graphs = [os.path.join(path, graphs) for path in os.listdir(graphs)]
    #
    # # graphs = {}
    # # train = pd.read_csv('./data/mol/train.csv')
    # # for _, row in tqdm(train.iterrows()):
    # #     if row['molecule_name'] not in graphs:
    # #         graphs[row['molecule_name']] = {'x': [], 'edge_index': [], 'edge_attr': [], 'y': []}
    # #
    # #     graph = graphs[row['molecule_name']]
    # #
    # #     graph['x'].append()
    #
    # graphs = [
    #
    # dataset = Dataset(graphs)
    #
    # print(dataset[0])


if __name__ == '__main__':
    main()
