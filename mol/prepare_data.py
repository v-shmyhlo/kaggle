from tqdm import tqdm
from functools import partial
import pandas as pd
from multiprocessing import Pool
import ase.io
import ase.data
import torch
import os
from torch_geometric.data import Data
import numpy as np

structures_path = './data/mol/structures'


def load_mol(name):
    return ase.io.read(os.path.join(structures_path, '{}.xyz'.format(name)))


def graph_to_data(pair, symbol_to_index, bond_to_index):
    mol_name, (nodes, edges) = pair

    num_nodes = len(nodes)
    num_edges = len(edges)

    # TODO: node mean stats

    # building x
    x = torch.zeros(num_nodes, 8)  # symbol_type, x, y, z, am, cr, gsmm, vdwr
    for node in nodes:
        x[node.index, 0] = symbol_to_index[node.symbol]
        x[node.index, 1:4] = torch.tensor([node.x, node.y, node.z])
        features = [
            ase.data.atomic_masses,
            ase.data.covalent_radii,
            ase.data.ground_state_magnetic_moments,
            ase.data.vdw_radii
        ]
        x[node.index, 4:] = torch.tensor([f[node.number] for f in features])

    # extracting fields
    i = edges['atom_index_0'].values
    j = edges['atom_index_1'].values
    bond = edges['type'].values
    coupling = edges[['scalar_coupling_constant', 'fc', 'sd', 'pso', 'dso']].values

    # building edge_index
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


def collect_edges(mol_name, edges):
    return edges[edges['molecule_name'] == mol_name]


def main():
    num_workers = os.cpu_count()
    mol_names = pd.read_csv('./data/mol/dipole_moments.csv')['molecule_name'].values

    with Pool(num_workers) as pool:
        nodes = pool.map(load_mol, tqdm(mol_names, desc='loading nodes'))

    edges = pd.read_csv('./data/mol/train.csv')
    contribs = pd.read_csv('./data/mol/scalar_coupling_contributions.csv')
    edges[['fc', 'sd', 'pso', 'dso']] = contribs[['fc', 'sd', 'pso', 'dso']]
    edges = edges.set_index('molecule_name')
    edges = [edges.loc[[mol_name]] for mol_name in tqdm(mol_names, desc='loading edges')]

    graphs = {mol_name: (ns, es) for mol_name, ns, es in zip(mol_names, nodes, edges)}

    symbol_to_index = set()
    bond_to_index = set()
    for nodes, edges in tqdm(graphs.values(), 'building mapping'):
        symbol_to_index.update(node.symbol for node in nodes)
        bond_to_index.update(edges['type'].values)

    symbol_to_index = {s: i for i, s in enumerate(sorted(symbol_to_index))}
    bond_to_index = {b: i for i, b in enumerate(sorted(bond_to_index))}
    print('symbol_to_index', symbol_to_index)
    print('bond_to_index', bond_to_index)

    with Pool(num_workers) as pool:
        pool.map(
            partial(graph_to_data, symbol_to_index=symbol_to_index, bond_to_index=bond_to_index),
            tqdm(graphs.items(), desc='saving data'))


if __name__ == '__main__':
    main()
