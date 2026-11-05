"""Linearized FEM/Newton loss utilities used during training."""

import math

import torch

from gnn_comsol.data.normalization import (
    PRESSURE_COLUMNS,
    VELOCITY_COLUMNS,
)


def fem_linearized_residual_metrics(
    velocity_preds,
    pressure_preds,
    velocity_target,
    pressure_target,
    jacobian_values,
    jacobian_row,
    jacobian_col,
    jacobian_shape,
    jacobian_dof_p,
    jacobian_dof_u,
    jacobian_dof_v,
    state_scale,
    topk_fraction=0.10,
):
    """
    Linearized COMSOL FEM residual around the converged target:

        r_lin = J(x*) @ (x - x*)

    Returns batch-mean diagnostics:

        mse   = mean(r_lin^2)

    """
    if not 0.0 < topk_fraction <= 1.0:
        raise ValueError(
            "topk_fraction must satisfy 0 < topk_fraction <= 1."
        )
    if velocity_preds.ndim != 3:
        raise ValueError(
            "velocity_preds must have shape (B, N, 2)."
        )

    if pressure_preds.ndim != 3:
        raise ValueError(
            "pressure_preds must have shape (B, N, 1)."
        )

    batch_size = velocity_preds.shape[0]
    num_nodes = velocity_preds.shape[1]

    expected_dofs = 3 * num_nodes

    if jacobian_shape != (
        expected_dofs,
        expected_dofs,
    ):
        raise ValueError(
            f"Jacobian shape {jacobian_shape} does not match "
            f"3*N = {expected_dofs}."
        )

    # Nondimensional error

    velocity_error_nd = (
        velocity_preds - velocity_target
    )

    pressure_error_nd = (
        pressure_preds - pressure_target
    )

    # Convert to physical units

    velocity_scale = state_scale[VELOCITY_COLUMNS]
    pressure_scale = state_scale[PRESSURE_COLUMNS]

    velocity_error = (
        velocity_error_nd * velocity_scale
    )

    pressure_error = (
        pressure_error_nd * pressure_scale
    )

    # Assemble FEM error:
    #
    # [p0,u0,v0,p1,u1,v1,...]

    fem_error = torch.zeros(
        (
            batch_size,
            expected_dofs,
        ),
        device=velocity_preds.device,
        dtype=velocity_preds.dtype,
    )

    fem_error[:, jacobian_dof_p] = (
        pressure_error.squeeze(-1)
    )

    fem_error[:, jacobian_dof_u] = (
        velocity_error[:, :, 0]
    )

    fem_error[:, jacobian_dof_v] = (
        velocity_error[:, :, 1]
    )

    mse_values = []
    topk_values = []

    # J(x*) @ (x - x*)

    indices = torch.stack(
        [
            jacobian_row,
            jacobian_col,
        ],
        dim=0,
    )

    for i in range(batch_size):

        J = torch.sparse_coo_tensor(
            indices,
            jacobian_values[i],
            size=jacobian_shape,
            device=velocity_preds.device,
            dtype=velocity_preds.dtype,
        ).coalesce()

        residual = torch.sparse.mm(
            J,
            fem_error[i].unsqueeze(1),
        ).squeeze(1)

        squared_residual = residual.square()

        mse_values.append(
            squared_residual.mean()
        )

        k = max(
            1,
            math.ceil(
                topk_fraction
                * squared_residual.numel()
            )
        )

        topk_values.append(
            torch.topk(
                squared_residual,
                k=k,
                largest=True,
                sorted=False,
            ).values.mean()
        )

    return {
        "mse": torch.stack(mse_values).mean(),
        "topk": torch.stack(topk_values).mean(),
    }

