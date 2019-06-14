from tqdm import tqdm
from functools import partial
import pandas as pd
from multiprocessing import Pool
import ase.io
import torch
import os
from torch_geometric.data import Data
import numpy as np

# # building edge_index
# i, j, d = ase.neighborlist.neighbor_list('ijd', nodes, ase.utils.natural_cutoffs(nodes))
# num_edges = len(i)
# edge_index = torch.tensor(np.stack([i, j], 0))

structures_path = './data/mol/structures'


def load_mol(name):
    return ase.io.read(os.path.join(structures_path, '{}.xyz'.format(name)))


def graph_to_data(pair, symbol_to_index, bond_to_index):
    mol_name, (nodes, edges) = pair

    num_nodes = len(nodes)
    num_edges = len(edges)

    # building x
    x = torch.zeros(num_nodes, 4)  # symbol_type, x, y, z
    for node in nodes:
        x[node.index, 0] = symbol_to_index[node.symbol]
        x[node.index, 1:] = torch.tensor([node.x, node.y, node.z])

    # building edge_index
    if len(edges) > 0:
        i, j, bond, coupling = map(list, zip(*edges))
    else:
        i, j, bond, coupling = [], [], [], []
    edge_index = torch.tensor(np.array([i, j]))

    # building edge_attr
    edge_attr = torch.zeros(num_edges, 5)  # bond_type, dist, x_dist, y_dist, z_dist
    edge_attr[:, 0] = torch.tensor([bond_to_index[b] for b in bond])
    dist = np.linalg.norm(nodes.positions[i] - nodes.positions[j], axis=-1)
    edge_attr[:, 1] = torch.tensor(dist)
    dist = np.abs(nodes.positions[i] - nodes.positions[j])
    edge_attr[:, 2:] = torch.tensor(dist)

    # building y
    y = torch.tensor(coupling)

    # building u
    u = [*nodes.positions.mean(0), *nodes.positions.std(0)]
    dist = np.linalg.norm(nodes.positions[i] - nodes.positions[j], axis=-1)
    u = [*u, dist.mean(), dist.std()]
    dist = np.abs(nodes.positions[i] - nodes.positions[j])
    u = [*u, *dist.mean(0), *dist.std(0)]
    u = torch.tensor([u])
    u[u != u] = 0.

    data = Data(
        x=x.float(),
        edge_index=edge_index.long(),
        edge_attr=edge_attr.float(),
        y=y.float(),
        u=u.float())

    path = './data/mol/graphs/{}.pth'.format(mol_name)
    torch.save(data, path)


def main():
    mol_names = sorted(os.path.splitext(path)[0] for path in os.listdir(structures_path))

    with Pool(os.cpu_count()) as pool:
        mols = pool.map(load_mol, tqdm(mol_names, desc='loading molecules'))
        graphs = {mol_name: (nodes, []) for mol_name, nodes in zip(mol_names, mols)}

    edges = pd.read_csv('./data/mol/train.csv')
    contribs = pd.read_csv('./data/mol/scalar_coupling_contributions.csv')
    edges[['fc', 'sd', 'pso', 'dso']] = contribs[['fc', 'sd', 'pso', 'dso']]
    # edge = [row[c] for c in
    #         ['atom_index_0', 'atom_index_1', 'type', 'scalar_coupling_constant', 'fc', 'sd', 'pso', 'dso']]
    # edges = edges.iloc[:1000]
    # for _, row in tqdm(edges.iterrows(), total=len(edges), desc='loading edges'):
    #     mol_name = row['molecule_name']
    #     graphs[mol_name][1].append(row)

    # graphs[mol]

    # for mol_name in mol_names:

    symbol_to_index = set()
    bond_to_index = set()
    for nodes, edges in tqdm(graphs.values(), 'building mapping'):
        for node in nodes:
            symbol_to_index.add(node.symbol)
        for _, _, bond, _ in edges:
            bond_to_index.add(bond)

    symbol_to_index = {s: i for i, s in enumerate(sorted(symbol_to_index))}
    bond_to_index = {b: i for i, b in enumerate(sorted(bond_to_index))}
    print('symbol_to_index', symbol_to_index)
    print('bond_to_index', bond_to_index)

    with Pool(os.cpu_count()) as pool:
        pool.map(
            partial(graph_to_data, symbol_to_index=symbol_to_index, bond_to_index=bond_to_index),
            tqdm(graphs.items(), desc='saving data'))


if __name__ == '__main__':
    main()
