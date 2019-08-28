import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_scatter import scatter_mean


class LinearNorm(nn.Sequential):
    def __init__(self, in_features, out_features):
        super().__init__(
            nn.Linear(in_features, out_features, bias=False),
            nn.BatchNorm1d(out_features))


class ReLU(nn.ReLU):
    def __init__(self):
        super().__init__(inplace=True)


class Model(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()

        features = 256
        self.project = LinearNorm(1280, features)
        self.layer_1 = gnn.MetaLayer(
            EdgeModel((features, 0, 0), features),
            NodeModel((features, features, 0), features),
            GlobalModel((features, features, 0), features))
        self.layer_2 = gnn.MetaLayer(
            EdgeModel((features, 0, 0), features),
            NodeModel((features, features, 0), features),
            GlobalModel((features, features, 0), features))
        self.layer_3 = gnn.MetaLayer(
            EdgeModel((features, 0, 0), features),
            NodeModel((features, features, 0), features),
            GlobalModel((features, features, 0), features))
        self.output = nn.Linear(features, num_classes)

    def forward(self, batch):
        x, u, batch = batch.x, batch.u, batch.batch

        # x = x.mean(1)
        x = self.project(x)

        # batch = batch.view(batch.size(0), 1).repeat(1, 2).view(batch.size(0) * 2)
        # x = x.view(x.size(0) * x.size(1), x.size(2))

        k = 10

        edge_index = gnn.knn_graph(x, k, batch, loop=False, flow='source_to_target')
        edge_attr = torch.zeros(edge_index.size(1), 0, device=x.device)
        u = torch.zeros(u.size(0), 0, device=x.device)
        x, _, _ = self.layer_1(x, edge_index, edge_attr, u, batch)
        edge_index = gnn.knn_graph(x, k, batch, loop=False, flow='source_to_target')
        edge_attr = torch.zeros(edge_index.size(1), 0, device=x.device)
        u = torch.zeros(u.size(0), 0, device=x.device)
        x, _, _ = self.layer_2(x, edge_index, edge_attr, u, batch)
        edge_index = gnn.knn_graph(x, k, batch, loop=False, flow='source_to_target')
        edge_attr = torch.zeros(edge_index.size(1), 0, device=x.device)
        u = torch.zeros(u.size(0), 0, device=x.device)
        x, _, _ = self.layer_3(x, edge_index, edge_attr, u, batch)

        logits = self.output(x)

        return logits


class EdgeModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.edge_layer = nn.Sequential(
            LinearNorm(in_features[0] * 2 + in_features[1] + in_features[2], out_features),
            ReLU(),
            LinearNorm(out_features, out_features))

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.

        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        return self.edge_layer(out)


class NodeModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.node_layer_1 = nn.Sequential(
            LinearNorm(in_features[0] + in_features[1], out_features),
            ReLU(),
            LinearNorm(out_features, out_features))
        self.node_layer_2 = nn.Sequential(
            LinearNorm(in_features[0] + out_features + in_features[2], out_features),
            ReLU(),
            LinearNorm(out_features, out_features))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        row, col = edge_index
        out = torch.cat([x[col], edge_attr], dim=1)
        out = self.node_layer_1(out)
        out = scatter_mean(out, row, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u[batch]], dim=1)
        return self.node_layer_2(out)


class GlobalModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.global_layer = nn.Sequential(
            LinearNorm(in_features[2] + in_features[0], out_features),
            ReLU(),
            LinearNorm(out_features, out_features))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        # out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        # return self.global_layer(out)

        return u
