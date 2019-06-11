from torch_geometric.datasets import TUDataset
from tqdm import tqdm

# dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

# from torch_geometric.datasets import Planetoid
#
# dataset = Planetoid(root='/tmp/Cora', name='Cora')
# print(dataset[0])

import pandas as pd
from torch_geometric.data import Data
from mol.dataset import Dataset

graphs = {}
train = pd.read_csv('./data/mol/train.csv')
for _, row in tqdm(train.iterrows()):
    if row['molecule_name'] not in graphs:
        graphs[row['molecule_name']] = {'x': [], 'edge_index': [], 'edge_attr': [], 'y': []}

    graph = graphs[row['molecule_name']]

    graph['x'].append()

graphs = [
    Data(
        x=graph['x'],
        edge_index=graph['edge_index'],
        edge_attr=graph['edge_attr'],
        y=graph['y']
    ) for graph in graphs.values()]

dataset = Dataset(graphs)

print(dataset[0])
