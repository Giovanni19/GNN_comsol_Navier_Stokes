"""The plain graph convolutional network."""

import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNNet(nn.Module):
    """
    A stack of GCNConv layers followed by a linear head.

    No residual connections and no normalization, so deep stacks
    over-smooth.
    """

    def __init__(
        self,
        num_in,
        num_out,
        num_neurons,
        num_layers,
        dropout
    ):

        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(GCNConv(num_in, num_neurons))

        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(num_neurons, num_neurons))

        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.out_layer = nn.Linear(num_neurons, num_out)

    def forward(self, graph):

        x = graph.x
        edge_index = graph.edge_index
        edge_weight = graph.edge_weight

        for layer in self.layers:

            x = layer(x, edge_index, edge_weight=edge_weight)
            x = self.activation(x)
            x = self.dropout(x)

        return self.out_layer(x)
