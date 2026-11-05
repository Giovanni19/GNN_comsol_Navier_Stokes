import torch
from torch_geometric.data import Data

from physicsnemo.datapipes.gnn.bsms import BistrideMultiLayerGraph


def build_bsms_hierarchy(
    edge_index,
    num_nodes,
    pos,
    num_mesh_levels,
):
    """Build the PyTorch BSMS hierarchy for one mesh."""

    # PhysicsNeMo's BSMS hierarchy builder internally converts graph.pos
    # to NumPy, so the hierarchy must be constructed on the CPU.
    graph = Data(
        edge_index=torch.as_tensor(
            edge_index,
            dtype=torch.long,
            device="cpu",
        ),
        pos=torch.as_tensor(
            pos,
            device="cpu",
        ),
        num_nodes=num_nodes,
    )

    multi_layer_graph = BistrideMultiLayerGraph(
        graph,
        num_mesh_levels,
    )

    _, flat_edges, pool_ids = (
        multi_layer_graph.get_multi_layer_graphs()
    )

    if len(flat_edges) != num_mesh_levels + 1:
        raise ValueError(
            f"Expected {num_mesh_levels + 1} BSMS edge levels, "
            f"got {len(flat_edges)}."
        )

    if len(pool_ids) != num_mesh_levels:
        raise ValueError(
            f"Expected {num_mesh_levels} pooling levels, "
            f"got {len(pool_ids)}."
        )

    return {
        "edge_indices": [
            torch.as_tensor(edges, dtype=torch.long)
            for edges in flat_edges
        ],
        "pool_indices": [
            torch.as_tensor(indices, dtype=torch.long)
            for indices in pool_ids
        ],
    }


def build_bsms_hierarchies(
    simulations,
    config,
):
    """
    Build the BSMS hierarchies required by each network.

    A separate hierarchy is built for velocity and pressure, so the two
    networks may use BSMS independently and may use different numbers
    of mesh levels.

    Returns
    -------
    dict
        hierarchies[network_name][simulation_id]
    """

    hierarchies = {}

    for network_name, network in config["networks"].items():

        if network["architecture"] != "bsms":
            continue

        num_mesh_levels = network["num_mesh_levels"]

        hierarchies[network_name] = {}

        for simulation in simulations:

            hierarchies[network_name][simulation.simulation_id] = (
                build_bsms_hierarchy(
                    simulation.edge_index,
                    simulation.num_nodes,
                    simulation.pos,
                    num_mesh_levels,
                )
            )

    return hierarchies