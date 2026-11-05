"""
NVIDIA PhysicsNeMo BiStrideMeshGraphNet, used for pressure.

The fine graph must carry x (N, F), edge_index (2, E),
edge_attr (E, 3) = [dx, dy, distance] and pos (N, 2). The multiscale
hierarchy (graph/bsms.py) is passed separately as ms_edges and ms_ids.
"""

import torch.nn as nn

from physicsnemo.models.meshgraphnet import BiStrideMeshGraphNet


class PressureBSMSGNN(nn.Module):

    def __init__(
        self,
        num_in,
        num_out=1,
        edge_dim=3,
        hidden_dim=128,
        processor_size=10,
        num_layers_node_processor=2,
        num_layers_edge_processor=2,
        num_mesh_levels=2,
        num_layers_bistride=2,
        bistride_unet_levels=1,
        pos_dim=2,
        mlp_activation_fn="relu",
        num_layers_node_encoder=2,
        num_layers_edge_encoder=2,
        num_layers_node_decoder=2,
        aggregation="sum",
        do_concat_trick=False,
        num_processor_checkpoint_segments=0,
        recompute_activation=False,
    ):
        super().__init__()

        self.model = BiStrideMeshGraphNet(
            input_dim_nodes=num_in,
            input_dim_edges=edge_dim,
            output_dim=num_out,

            processor_size=processor_size,
            mlp_activation_fn=mlp_activation_fn,

            num_layers_node_processor=num_layers_node_processor,
            num_layers_edge_processor=num_layers_edge_processor,

            num_mesh_levels=num_mesh_levels,

            bistride_pos_dim=pos_dim,
            num_layers_bistride=num_layers_bistride,
            bistride_unet_levels=bistride_unet_levels,

            hidden_dim_processor=hidden_dim,

            hidden_dim_node_encoder=hidden_dim,
            num_layers_node_encoder=num_layers_node_encoder,

            hidden_dim_edge_encoder=hidden_dim,
            num_layers_edge_encoder=num_layers_edge_encoder,

            hidden_dim_node_decoder=hidden_dim,
            num_layers_node_decoder=num_layers_node_decoder,

            aggregation=aggregation,

            do_concat_trick=do_concat_trick,
            num_processor_checkpoint_segments=(
                num_processor_checkpoint_segments
            ),
            recompute_activation=recompute_activation,
        )

    def forward(
        self,
        graph,
        ms_edges,
        ms_ids,
    ):
        if not hasattr(graph, "edge_attr"):
            raise ValueError(
                "BiStrideMeshGraphNet requires graph.edge_attr. "
                "Expected [dx, dy, distance]."
            )

        if not hasattr(graph, "pos"):
            raise ValueError(
                "BiStrideMeshGraphNet requires graph.pos."
            )

        return self.model(
            node_features=graph.x,
            edge_features=graph.edge_attr,
            graph=graph,
            ms_edges=ms_edges,
            ms_ids=ms_ids,
        )
