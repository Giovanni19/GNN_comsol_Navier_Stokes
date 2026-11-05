"""
Hard imposition of the Dirichlet boundary conditions: the prediction is replaced by the value prescribed by the COMSOL
model, stored in the dataset.

Used both inside the loss and at inference, so the two always agree. 
"""

import numpy as np
import torch


def _checked(velocity, dirichlet_mask, velocity_bc):
    """Shape checks: a mismatch would silently broadcast."""

    if velocity.ndim != 2 or velocity.shape[1] != 2:
        raise ValueError(
            "velocity must have shape (nodes, 2), got "
            f"{tuple(velocity.shape)}."
        )

    if velocity_bc.shape != velocity.shape:
        raise ValueError(
            f"velocity_bc has shape {tuple(velocity_bc.shape)} but the "
            f"velocity has {tuple(velocity.shape)}: they must describe "
            "the same nodes."
        )

    if dirichlet_mask.ndim != 1:
        raise ValueError(
            "dirichlet_mask must have shape (nodes,), got "
            f"{tuple(dirichlet_mask.shape)}."
        )

    if len(dirichlet_mask) != len(velocity):
        raise ValueError(
            f"dirichlet_mask covers {len(dirichlet_mask)} nodes but "
            f"the velocity has {len(velocity)}."
        )

    if dirichlet_mask.dtype != torch.bool:
        dirichlet_mask = dirichlet_mask.bool()

    return dirichlet_mask


def _checked_pressure(pressure, dirichlet_mask, pressure_bc):
    """Shape checks for the pressure Dirichlet values."""

    if pressure.ndim != 1:
        raise ValueError(
            "pressure must have shape (nodes,), got "
            f"{tuple(pressure.shape)}."
        )

    if pressure_bc.shape != pressure.shape:
        raise ValueError(
            f"pressure_bc has shape {tuple(pressure_bc.shape)} but the "
            f"pressure has {tuple(pressure.shape)}: they must describe "
            "the same nodes."
        )

    if dirichlet_mask.ndim != 1:
        raise ValueError(
            "dirichlet_mask must have shape (nodes,), got "
            f"{tuple(dirichlet_mask.shape)}."
        )

    if len(dirichlet_mask) != len(pressure):
        raise ValueError(
            f"dirichlet_mask covers {len(dirichlet_mask)} nodes but "
            f"the pressure has {len(pressure)}."
        )

    if dirichlet_mask.dtype != torch.bool:
        dirichlet_mask = dirichlet_mask.bool()

    return dirichlet_mask


def apply_velocity_dirichlet(velocity, dirichlet_mask, velocity_bc):
    """
    Replace the predicted velocity at Dirichlet nodes with the
    prescribed one.

    Parameters
    ----------
    velocity : torch.Tensor, shape (N, 2)
        Predicted velocity at the mesh nodes, in PHYSICAL units.

    dirichlet_mask : torch.Tensor, shape (N,)
        True where the velocity is prescribed: walls and inlet, not the
        outlet.

    velocity_bc : torch.Tensor, shape (N, 2)
        The prescribed values; only the masked rows are read.

    Returns
    -------
    torch.Tensor, shape (N, 2)

    The Dirichlet nodes receive no gradient (`torch.where`).
    """

    dirichlet_mask = _checked(velocity, dirichlet_mask, velocity_bc)

    return torch.where(
        dirichlet_mask.unsqueeze(1),
        velocity_bc,
        velocity,
    )


def apply_pressure_dirichlet(
    pressure,
    dirichlet_mask,
    pressure_bc,
):
    """
    Replace the predicted pressure at Dirichlet nodes with the
    prescribed pressure.

    Parameters
    ----------
    pressure : torch.Tensor, shape (N,)
        Predicted pressure at the mesh nodes, in PHYSICAL units.

    dirichlet_mask : torch.Tensor, shape (N,)
        True where pressure is prescribed (the outlet).

    pressure_bc : torch.Tensor, shape (N,)
        The prescribed values; only the masked entries are read.

    Returns
    -------
    torch.Tensor, shape (N,)

    The Dirichlet nodes receive no gradient (`torch.where`).
    """

    dirichlet_mask = _checked_pressure(
        pressure,
        dirichlet_mask,
        pressure_bc,
    )

    return torch.where(
        dirichlet_mask,
        pressure_bc,
        pressure,
    )


def velocity_dirichlet_violation(velocity, dirichlet_mask, velocity_bc):
    """
    How far the prediction is from the boundary condition, over the
    Dirichlet nodes only: how much work the hard imposition does.

    Returns
    -------
    dict with "rms", "max" (physical units, NaN when the mask is empty)
    and "num_nodes".
    """

    dirichlet_mask = _checked(velocity, dirichlet_mask, velocity_bc)

    if not bool(dirichlet_mask.any()):
        return {
            "rms": float("nan"),
            "max": float("nan"),
            "num_nodes": 0,
        }

    error = (
        velocity[dirichlet_mask] - velocity_bc[dirichlet_mask]
    )

    return {
        "rms": float(error.square().mean().sqrt()),
        "max": float(error.abs().max()),
        "num_nodes": int(dirichlet_mask.sum()),
    }


def boundary_consistency(velocity_bc_target, dirichlet_mask, velocity_target):
    """
    Do the prescribed boundary condition and the COMSOL solution agree
    where the condition is prescribed?

    They must agree to export precision; if not, the dataset is wrong,
    usually at a corner node that belongs to a wall and to the inlet
    (first_database.m forces those to no-slip).

    Parameters
    ----------
    velocity_bc_target : ndarray, shape (S, N, 2)
    dirichlet_mask : ndarray, shape (N,)
    velocity_target : ndarray, shape (S, N, 2)
        The solved velocity at the SAME times as the prescribed values.

    Returns
    -------
    dict with "largest" (the worst absolute disagreement), "scale" (the
    largest velocity magnitude in the field, to read it against),
    "relative" and "num_nodes".
    """

    mask = np.asarray(dirichlet_mask).astype(bool)

    prescribed = np.asarray(velocity_bc_target)[:, mask, :]
    solved = np.asarray(velocity_target)[:, mask, :]

    if prescribed.shape != solved.shape:
        raise ValueError(
            f"The prescribed values have shape {prescribed.shape} and "
            f"the solved field {solved.shape}: they must cover the same "
            "samples and the same nodes."
        )

    largest = (
        float(np.abs(prescribed - solved).max())
        if prescribed.size
        else 0.0
    )

    scale = float(np.abs(velocity_target).max())

    return {
        "largest": largest,
        "scale": scale,
        "relative": largest / scale if scale > 0 else float("nan"),
        "num_nodes": int(mask.sum()),
    }

def apply_velocity_dirichlet_to_delta(
    velocity_delta,
    velocity_now,
    dirichlet_mask,
    velocity_bc,
    velocity_scale,
):
    """
    Hard-impose velocity Dirichlet BCs directly on a normalized
    predicted increment.

    The network representation is

        delta_u* = (u_{n+1} - u_n) / U_inf

    so at Dirichlet nodes the required increment is

        delta_u*_BC = (u_BC - u_n) / U_inf.

    Parameters
    ----------
    velocity_delta : torch.Tensor, shape (N, 2)
        Predicted nondimensional velocity increment.

    velocity_now : torch.Tensor, shape (N, 2)
        Current velocity in PHYSICAL units.

    dirichlet_mask : torch.Tensor, shape (N,)
        True at velocity Dirichlet nodes.

    velocity_bc : torch.Tensor, shape (N, 2)
        Prescribed next-step velocity in PHYSICAL units.

    velocity_scale : torch.Tensor, shape (2,)
        Physical velocity scale, normally [U_inf, U_inf].

    Returns
    -------
    torch.Tensor, shape (N, 2)
        Nondimensional increment satisfying the BC exactly.
    """

    dirichlet_mask = _checked(
        velocity_now,
        dirichlet_mask,
        velocity_bc,
    )

    if velocity_delta.shape != velocity_now.shape:
        raise ValueError(
            f"velocity_delta has shape {tuple(velocity_delta.shape)} "
            f"but velocity_now has shape {tuple(velocity_now.shape)}."
        )

    required_delta = (
        velocity_bc - velocity_now
    ) / velocity_scale

    return torch.where(
        dirichlet_mask.unsqueeze(1),
        required_delta,
        velocity_delta,
    )


def apply_pressure_dirichlet_to_delta(
    pressure_delta,
    pressure_now,
    dirichlet_mask,
    pressure_bc,
    pressure_scale,
):
    """
    Hard-impose pressure Dirichlet BCs directly on a normalized
    predicted increment.

    The network representation is

        delta_p* = (p_{n+1} - p_n) / (rho U_inf^2)

    so at Dirichlet nodes the required increment is

        delta_p*_BC = (p_BC - p_n) / (rho U_inf^2).
    """

    dirichlet_mask = _checked_pressure(
        pressure_now,
        dirichlet_mask,
        pressure_bc,
    )

    if pressure_delta.ndim == 2 and pressure_delta.shape[1] == 1:
        pressure_delta_1d = pressure_delta.squeeze(-1)
    elif pressure_delta.ndim == 1:
        pressure_delta_1d = pressure_delta
    else:
        raise ValueError(
            "pressure_delta must have shape (N,) or (N, 1), got "
            f"{tuple(pressure_delta.shape)}."
        )

    required_delta = (
        pressure_bc - pressure_now
    ) / pressure_scale.squeeze()

    corrected = torch.where(
        dirichlet_mask,
        required_delta,
        pressure_delta_1d,
    )

    return corrected.unsqueeze(-1)