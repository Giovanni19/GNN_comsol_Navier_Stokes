"""
NVIDIA PhysicsNeMo MeshGraphNet, behind the `model(graph)` interface of
the other models.

The graph must carry x (N, F), edge_index (2, E) and
edge_attr (E, 3) = [dx, dy, distance].
"""

import torch.nn as nn

from physicsnemo.models.meshgraphnet import MeshGraphNet


class MeshGraphNetModel(nn.Module):
    """
    hidden_dim is the latent size of encoders, processor and decoder;
    processor_size is the number of message-passing blocks.
    """

    def __init__(
        self,
        num_in,
        num_out,
        edge_dim=3,
        hidden_dim=128,
        processor_size=15,
        mlp_activation_fn="relu",
        num_layers_node_processor=2,
        num_layers_edge_processor=2,
        num_layers_node_encoder=2,
        num_layers_edge_encoder=2,
        num_layers_node_decoder=2,
        aggregation="sum",
    ):

        super().__init__()

        self.model = MeshGraphNet(
            input_dim_nodes=num_in,
            input_dim_edges=edge_dim,
            output_dim=num_out,

            processor_size=processor_size,

            mlp_activation_fn=mlp_activation_fn,

            num_layers_node_processor=num_layers_node_processor,
            num_layers_edge_processor=num_layers_edge_processor,

            hidden_dim_processor=hidden_dim,

            hidden_dim_node_encoder=hidden_dim,
            num_layers_node_encoder=num_layers_node_encoder,

            hidden_dim_edge_encoder=hidden_dim,
            num_layers_edge_encoder=num_layers_edge_encoder,

            hidden_dim_node_decoder=hidden_dim,
            num_layers_node_decoder=num_layers_node_decoder,

            aggregation=aggregation,
        )

    def forward(self, graph):

        if not hasattr(graph, "edge_attr"):
            raise ValueError(
                "MeshGraphNet requires graph.edge_attr. "
                "Expected geometric edge features "
                "[dx, dy, distance]."
            )

        return self.model(
            node_features=graph.x,
            edge_features=graph.edge_attr,
            graph=graph,
        )
