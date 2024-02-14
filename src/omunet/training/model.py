import torch
from dgl.nn import pytorch as dglnn
from torch import nn
from torch.nn import functional as F


class HeteroDotProductPredictor(nn.Module):
    def forward(
        self,
        edges_supervised,
        h_first,
        h_second,
    ):
        h_first = h_first[edges_supervised[0]]
        h_second = h_second[edges_supervised[1]]
        h_src_nodes = h_first / h_first.norm(dim=1, p=2).reshape(h_first.shape[0], -1)
        h_tgt_nodes = h_second / h_second.norm(dim=1, p=2).reshape(
            h_second.shape[0], -1
        )

        # row wise dot product
        return torch.sum(h_src_nodes * h_tgt_nodes, dim=-1)


class RGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, amount_convolutions):
        super().__init__()
        self.convolutions = [
            dglnn.GraphConv(in_dim if i == 0 else hid_dim, hid_dim)
            for i in range(amount_convolutions)
        ]

    def forward(self, graph):
        h = graph.ndata["feat"]

        skips = [h]
        for convolution in self.convolutions:
            h = convolution(graph, h)
            h = F.relu(h)
            skips.append(h)

        return torch.hstack(skips)


class Model(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        amount_convolutions,
    ):
        super().__init__()
        self.rgcn = RGCN(in_dim, hidden_dim, amount_convolutions)
        self.pred = HeteroDotProductPredictor()

        input_dim_postprocess = hidden_dim * amount_convolutions + in_dim
        self.postprocess_first = nn.Linear(input_dim_postprocess, out_dim)
        self.postprocess_second = nn.Linear(input_dim_postprocess, out_dim)

    def forward(
        self,
        first_graph,
        second_graph,
        edges_positive_supervision,
        edges_negative_supervision,
    ):
        h_first = self.postprocess_first(self.rgcn(first_graph))
        h_second = self.postprocess_second(self.rgcn(second_graph))

        positive_scores = self.pred(edges_positive_supervision, h_first, h_second)
        negative_scores = self.pred(edges_negative_supervision, h_first, h_second)

        # sanity_check
        assert h_first.shape[0] == first_graph.num_nodes()
        assert h_second.shape[0] == second_graph.num_nodes()

        return (
            positive_scores.reshape(positive_scores.shape[0], -1),
            negative_scores.reshape(positive_scores.shape[0], -1),
            h_first,
            h_second,
        )
