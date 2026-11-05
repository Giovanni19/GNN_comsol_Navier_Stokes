"""
Convert normalized simulation arrays into the coupled training dataset.

A TensorDataset per simulation: the mesh topology is stored separately
and shared by all its transitions.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset


def to_tensor(array, dtype=torch.float32):
    """Convert NumPy-like data to a tensor without reconverting tensors."""
    if isinstance(array, torch.Tensor):
        return array
    return torch.as_tensor(np.asarray(array), dtype=dtype)


def _check_samples(name, array, num_samples):
    if len(array) != num_samples:
        raise ValueError(
            f"{name} has {len(array)} samples, expected {num_samples}."
        )


def _prepare_mask(mask, name, num_nodes):
    if mask is None:
        return None

    mask = to_tensor(mask, dtype=torch.bool).reshape(-1).bool()

    if len(mask) != num_nodes:
        raise ValueError(
            f"{name} has {len(mask)} nodes, expected {num_nodes}."
        )

    return mask


# Coupled velocity-pressure dataset

def create_coupled_dataset(
    velocity_X,
    pressure_X,
    velocity_Y,
    pressure_Y,
    simulation_id,
    delta_t,
    velocity_bc_target=None,
    velocity_dirichlet_mask=None,
    pressure_bc_target=None,
    pressure_dirichlet_mask=None,
    jacobian_values=None,
):
    """
    Build synchronized velocity-pressure samples.

    One item corresponds to one physical transition and contains:
        velocity input,
        pressure input,
        velocity target,
        pressure target,
        physical dt,
        simulation id,
        velocity BC + mask,
        pressure BC + mask,
        FEM Jacobian values.
    """

    velocity_X = to_tensor(velocity_X)
    pressure_X = to_tensor(pressure_X)
    velocity_Y = to_tensor(velocity_Y)
    pressure_Y = to_tensor(pressure_Y)
    delta_t = to_tensor(delta_t)

    num_samples = len(velocity_Y)
    num_nodes = velocity_X.shape[1]

    for name, array in (
        ("velocity_X", velocity_X),
        ("pressure_X", pressure_X),
        ("pressure_Y", pressure_Y),
        ("delta_t", delta_t),
    ):
        _check_samples(name, array, num_samples)

    for name, array in (
        ("pressure_X", pressure_X),
        ("velocity_Y", velocity_Y),
        ("pressure_Y", pressure_Y),
    ):
        if array.shape[1] != num_nodes:
            raise ValueError(
                f"{name} has {array.shape[1]} nodes, "
                f"expected {num_nodes}."
            )

    # Boundary conditions

    if velocity_bc_target is None:
        raise ValueError(
            "Coupled training requires velocity_bc_target."
        )

    velocity_bc_target = to_tensor(velocity_bc_target)

    if velocity_bc_target.shape != (num_samples, num_nodes, 2):
        raise ValueError(
            "velocity_bc_target has shape "
            f"{tuple(velocity_bc_target.shape)}, expected "
            f"({num_samples}, {num_nodes}, 2)."
        )

    if pressure_bc_target is None:
        raise ValueError(
            "Coupled training requires pressure_bc_target."
        )

    pressure_bc_target = to_tensor(pressure_bc_target)

    if pressure_bc_target.shape != (num_samples, num_nodes):
        raise ValueError(
            "pressure_bc_target has shape "
            f"{tuple(pressure_bc_target.shape)}, expected "
            f"({num_samples}, {num_nodes})."
        )

    if velocity_dirichlet_mask is None:
        raise ValueError(
            "Coupled training requires velocity_dirichlet_mask."
        )

    if pressure_dirichlet_mask is None:
        raise ValueError(
            "Coupled training requires pressure_dirichlet_mask."
        )

    velocity_dirichlet_mask = _prepare_mask(
        velocity_dirichlet_mask,
        "velocity_dirichlet_mask",
        num_nodes,
    )

    pressure_dirichlet_mask = _prepare_mask(
        pressure_dirichlet_mask,
        "pressure_dirichlet_mask",
        num_nodes,
    )

    # FEM Jacobian

    if jacobian_values is not None:

        jacobian_values = to_tensor(jacobian_values)

        if jacobian_values.ndim != 2:
            raise ValueError(
                "jacobian_values must have shape (S, nnz), got "
                f"{tuple(jacobian_values.shape)}."
            )

        _check_samples(
            "jacobian_values",
            jacobian_values,
            num_samples,
        )

        if not torch.isfinite(jacobian_values).all():
            raise ValueError(
                "jacobian_values contains NaN or Inf."
            )

    else:
        # Keep the coupled batch structure fixed at 11 entries.
        # With Newton loss disabled this tensor is never used.
        jacobian_values = torch.empty(
            (num_samples, 0),
            dtype=torch.float32,
        )

    # Static metadata repeated over transitions

    simulation_ids = torch.full(
        (num_samples,),
        int(simulation_id),
        dtype=torch.long,
    )

    velocity_mask = (
        velocity_dirichlet_mask
        .unsqueeze(0)
        .expand(num_samples, -1)
    )

    pressure_mask = (
        pressure_dirichlet_mask
        .unsqueeze(0)
        .expand(num_samples, -1)
    )

    return TensorDataset(
        velocity_X,
        pressure_X,
        velocity_Y,
        pressure_Y,
        delta_t,
        simulation_ids,
        velocity_bc_target,
        velocity_mask,
        pressure_bc_target,
        pressure_mask,
        jacobian_values,
    )

def build_graph_topologies(simulations):
    """
    Static fine-mesh topology of each simulation: edge_index,
    edge_weight (GCN), edge_attr = [dx, dy, distance] (MeshGraphNet),
    pos, and the FEM Jacobian pattern and DOF mapping.
    """

    graph_topologies = {}

    for simulation in simulations:

        simulation_id = simulation.simulation_id

        edge_index = torch.as_tensor(
            simulation.edge_index,
            dtype=torch.long,
        )

        edge_weight = torch.as_tensor(
            simulation.edge_weight,
            dtype=torch.float32,
        )

        pos = torch.as_tensor(
            simulation.pos,
            dtype=torch.float32,
        )

        # MeshGraphNet edge features

        source = edge_index[0]
        target = edge_index[1]

        relative_position = pos[target] - pos[source]

        distance = torch.linalg.vector_norm(
            relative_position,
            dim=1,
            keepdim=True,
        )

        edge_attr = torch.cat(
            [
                relative_position,
                distance,
            ],
            dim=1,
        )

        graph_topologies[simulation_id] = {
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "edge_attr": edge_attr,
            "pos": pos,

            # Only jacobian_values changes with time; the rest is here.
            "jacobian_row": (
                None
                if simulation.jacobian_row is None
                else torch.as_tensor(
                    simulation.jacobian_row,
                    dtype=torch.long,
                )
            ),

            "jacobian_col": (
                None
                if simulation.jacobian_col is None
                else torch.as_tensor(
                    simulation.jacobian_col,
                    dtype=torch.long,
                )
            ),

            "jacobian_shape": (
                None
                if simulation.jacobian_shape is None
                else tuple(
                    int(x)
                    for x in simulation.jacobian_shape
                )
            ),

            "jacobian_dof_p": (
                None
                if simulation.jacobian_dof_p is None
                else torch.as_tensor(
                    simulation.jacobian_dof_p,
                    dtype=torch.long,
                )
            ),

            "jacobian_dof_u": (
                None
                if simulation.jacobian_dof_u is None
                else torch.as_tensor(
                    simulation.jacobian_dof_u,
                    dtype=torch.long,
                )
            ),

            "jacobian_dof_v": (
                None
                if simulation.jacobian_dof_v is None
                else torch.as_tensor(
                    simulation.jacobian_dof_v,
                    dtype=torch.long,
                )
            ),
        }

    return graph_topologies