import torch
import torch_geometric

NUM_CLASSES = 1108

EXPS = [
    'HEPG2-01', 'HEPG2-02', 'HEPG2-03', 'HEPG2-04', 'HEPG2-05', 'HEPG2-06', 'HEPG2-07', 'HEPG2-08', 'HEPG2-09',
    'HEPG2-10', 'HEPG2-11', 'HUVEC-01', 'HUVEC-02', 'HUVEC-03', 'HUVEC-04', 'HUVEC-05', 'HUVEC-06', 'HUVEC-07',
    'HUVEC-08', 'HUVEC-09', 'HUVEC-10', 'HUVEC-11', 'HUVEC-12', 'HUVEC-13', 'HUVEC-14', 'HUVEC-15', 'HUVEC-16',
    'HUVEC-17', 'HUVEC-18', 'HUVEC-19', 'HUVEC-20', 'HUVEC-21', 'HUVEC-22', 'HUVEC-23', 'HUVEC-24', 'RPE-01',
    'RPE-02', 'RPE-03', 'RPE-04', 'RPE-05', 'RPE-06', 'RPE-07', 'RPE-08', 'RPE-09', 'RPE-10', 'RPE-11', 'U2OS-01',
    'U2OS-02', 'U2OS-03', 'U2OS-04', 'U2OS-05'
]


class TrainEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data, embs, transform=None):
        self.data = list(data.groupby('experiment'))
        self.embs = embs

        self.transform = transform
        self.cell_type_to_id = {cell_type: i for i, cell_type in enumerate(['HEPG2', 'HUVEC', 'RPE', 'U2OS'])}
        self.exp_to_id = {exp: i for i, exp in enumerate(EXPS)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        exp, group = self.data[item]
        cell_type, _ = exp.split('-')
        feat = [self.cell_type_to_id[cell_type]]

        input = build_graph(
            group,
            x=self.embs,
            y=group['sirna'].values,
            u=feat,
            exps=group['experiment'].apply(lambda exp: self.exp_to_id[exp]).values)

        if self.transform is not None:
            input = self.transform(input)

        return input


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data, embs, transform=None):
        self.data = list(data.groupby('experiment'))
        self.embs = embs

        self.transform = transform
        self.cell_type_to_id = {cell_type: i for i, cell_type in enumerate(['HEPG2', 'HUVEC', 'RPE', 'U2OS'])}
        self.exp_to_id = {exp: i for i, exp in enumerate(EXPS)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        exp, group = self.data[item]
        cell_type, _ = exp.split('-')
        feat = [self.cell_type_to_id[cell_type]]

        input = build_graph(
            group,
            x=self.embs,
            y=group['sirna'].values,
            u=feat,
            exps=group['experiment'].apply(lambda exp: self.exp_to_id[exp]).values)

        if self.transform is not None:
            input = self.transform(input)

        return input


def build_graph(group, x, y, u, exps):
    x = torch.tensor([x[id] for id in group['id_code']], dtype=torch.float)
    edge_index = None
    edge_attr = None
    y = torch.tensor(y, dtype=torch.long)
    u = torch.tensor(u, dtype=torch.long)
    exps = torch.tensor(exps, dtype=torch.long)

    graph = torch_geometric.data.Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        u=u,
        exps=exps)

    return graph
