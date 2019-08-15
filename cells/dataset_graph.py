import torch
import torch_geometric

NUM_CLASSES = 1108


class TrainEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data, embs, transform=None):
        self.data = list(data.groupby('experiment'))
        self.embs = embs

        self.transform = transform
        self.cell_type_to_id = {cell_type: i for i, cell_type in enumerate(['HEPG2', 'HUVEC', 'RPE', 'U2OS'])}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        exp, group = self.data[item]
        cell_type, _ = exp.split('-')
        feat = [self.cell_type_to_id[cell_type]]

        graph = build_graph(
            group,
            x=self.embs,
            y=group['sirna'].values,
            u=feat)

        # print(exp, group.shape)
        # fail
        #
        # image = []
        # for s in [1, 2]:
        #     image.extend(load_image(row['root'], row['experiment'], row['plate'], row['well'], s))
        #

        input = graph

        if self.transform is not None:
            input = self.transform(input)

        return input


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.cell_type_to_id = {cell_type: i for i, cell_type in enumerate(['HEPG2', 'HUVEC', 'RPE', 'U2OS'])}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        row = self.data.iloc[item]

        image = []
        for s in [1, 2]:
            image.extend(load_image(row['root'], row['experiment'], row['plate'], row['well'], s))

        cell_type = row['experiment'].split('-')[0]
        feat = torch.tensor([self.cell_type_to_id[cell_type], row['plate'] - 1])

        input = {
            'image': image,
            'feat': feat,
            'exp': row['experiment'],
            'plate': row['plate'],
            'id': row['id_code']
        }

        if self.transform is not None:
            input = self.transform(input)

        return input


def build_graph(group, x, y, u):
    x = torch.tensor([x[id] for id in group['id_code']], dtype=torch.float)
    edge_index = None
    edge_attr = None
    u = torch.tensor(u, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)

    graph = torch_geometric.data.Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        u=u)

    return graph
