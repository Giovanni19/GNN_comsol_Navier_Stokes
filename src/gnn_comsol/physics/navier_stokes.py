"""
Residuals of the incompressible Navier-Stokes equations on the mesh.

The discretization follows the Gen-FVGN paper (Li et al., 2024): the
variables live at the mesh vertices, the WLSQ gradient is reconstructed
there, and the equations are written on the cell-centered control
volumes, where the vertex quantities are interpolated with Eq. (A.3).

    rho (du/dt + (u.grad) u) = -grad p + mu laplacian(u)
    div(u) = 0

Both residuals are volume-integrated over a cell (Eq. 16-17), so they
carry a factor of the cell area and small cells weigh less.

One gradient reconstruction pass is enough: the viscous term takes the
divergence of the same gradient over the cell faces.
"""

import torch
import numpy as np
from .mesh import divergence_over_faces, interpolate_to_cell
from .wlsq import node_gradient_to_cell, wlsq_gradient


def continuity_residual(u, v, operators, geometry):
    """
    The incompressible continuity residual, integrated over each cell:

        R = (du/dx + dv/dy) * area

    Parameters
    ----------
    u, v : torch.Tensor, shape (N,)
        Velocity components at the mesh nodes, in physical units.

    operators : WLSQOperators
        Gradient operators of this mesh, see physics.wlsq.

    geometry : CellGeometry
        Control volumes of this mesh, see physics.mesh.

    Returns
    -------
    residual : torch.Tensor, shape (Nc,)
    """

    # u and v are reconstructed in a single pair of sparse products:
    # gradient[:, 0] is grad(u) and gradient[:, 1] is grad(v).
    velocity = torch.stack((u, v), dim=1)

    gradient_cell = node_gradient_to_cell(
        wlsq_gradient(velocity, operators),
        geometry.cell_index,
    )

    divergence = (
        gradient_cell[:, 0, 0]
        + gradient_cell[:, 1, 1]
    )

    return divergence * geometry.area


def continuity_loss(u, v, operators, geometry):
    """Mean squared incompressibility residual."""

    residual = continuity_residual(u, v, operators, geometry)

    return torch.mean(residual.square())


def momentum_terms(
    velocity_now,
    velocity_next,
    pressure_next,
    delta_t,
    fluid,
    operators,
    geometry,
):
    """
    Fully implicit momentum terms.

    Spatial terms are evaluated entirely at the predicted next state:

        rho (u^{n+1} - u^n) / dt
        + rho (u^{n+1} . grad) u^{n+1}
        + grad p^{n+1}
        - mu laplacian(u^{n+1})

    All terms are integrated over the control volume.
    """

    # Columns: u^n, v^n, u^{n+1}, v^{n+1}, p^{n+1}

    fields = torch.cat(
        (
            velocity_now,
            velocity_next,
            pressure_next.unsqueeze(1),
        ),
        dim=1,
    )

    gradients = wlsq_gradient(
        fields,
        operators,
    )

    values_cell = interpolate_to_cell(
        fields,
        gradients,
        geometry,
    )

    velocity_now_cell = values_cell[:, 0:2]
    velocity_next_cell = values_cell[:, 2:4]

    # Fully implicit: grad(u^{n+1}) only, not the midpoint.

    gradient_next = gradients[:, 2:4]

    gradient_next_cell = node_gradient_to_cell(
        gradient_next,
        geometry.cell_index,
    )

    area = geometry.area.unsqueeze(1)

    return {
        "time": (
            fluid.rho
            * (
                velocity_next_cell
                - velocity_now_cell
            )
            / delta_t
            * area
        ),

        "convection": (
            fluid.rho
            * (
                gradient_next_cell
                * velocity_next_cell.unsqueeze(1)
            ).sum(dim=-1)
            * area
        ),

        "pressure": (
            node_gradient_to_cell(
                gradients[:, 4],
                geometry.cell_index,
            )
            * area
        ),

        "viscous": (
            -fluid.mu
            * divergence_over_faces(
                gradient_next,
                geometry,
            )
        ),
    }


def momentum_residual(
    velocity_now,
    velocity_next,
    pressure_next,
    delta_t,
    fluid,
    operators,
    geometry,
):
    """
    The momentum residual of one transition, integrated over each cell.

        R = rho (u^{n+1}_c - u^n_c) / dt * area
          + rho (u^{n+1}_c . grad) u^{n+1}_c * area
          + grad p^{n+1}_c * area
          - mu sum_f (grad u^{n+1}_f . n) dl

    Parameters
    ----------
    velocity_now, velocity_next : torch.Tensor, shape (N, 2)
        Velocity at the nodes at t and t + dt, in physical units.

    pressure_next : torch.Tensor, shape (N,)
        Pressure at t + dt, in physical units.

    delta_t : float or 0-d torch.Tensor
        The PHYSICAL duration of the transition, in seconds - not the
        normalized dt fed to the network.

    fluid : FluidProperties

    operators : WLSQOperators
    geometry : CellGeometry

    Returns
    -------
    residual : torch.Tensor, shape (Nc, 2)
        Column 0 is the x-momentum residual, column 1 the y-momentum.
    Notes
    -----
    The step is fully implicit (backward Euler): convection, pressure
    and viscous terms are all evaluated at t + dt, not at the IMEX
    midpoint of Eq. (12) of the paper. See momentum_terms.
    """

    terms = momentum_terms(
        velocity_now,
        velocity_next,
        pressure_next,
        delta_t,
        fluid,
        operators,
        geometry,
    )

    return sum(terms.values())


def momentum_loss(
    velocity_now,
    velocity_next,
    pressure_next,
    delta_t,
    fluid,
    operators,
    geometry,
):
    """Mean squared momentum residual, over cells and both components."""

    residual = momentum_residual(
        velocity_now,
        velocity_next,
        pressure_next,
        delta_t,
        fluid,
        operators,
        geometry,
    )

    return torch.mean(residual.square())

def compute_physics_reference_scales(
    train_simulations,
    physics_geometries,
):
    """
    Mean continuity and momentum loss of the COMSOL ground truth over
    the training transitions.

    These values are used as reference scales for normalizing the
    coupled physics-loss terms.
    """

    continuity_values = []
    momentum_values = []

    print("\nComputing physics reference scales from TRAINING data:")

    for simulation in train_simulations:

        simulation_id = simulation.simulation_id

        if simulation_id not in physics_geometries:
            raise ValueError(
                f"Simulation {simulation_id} has no physics geometry."
            )

        physics_geometry = physics_geometries[simulation_id]

        device = torch.device("cpu")
        dtype = torch.float64

        operators = physics_geometry.operators(
            device=device,
            dtype=dtype,
        )

        cells = physics_geometry.cells(
            device=device,
            dtype=dtype,
        )

        fluid = physics_geometry.require_fluid()

        simulation_continuity = []
        simulation_momentum = []

        for transition in range(simulation.num_samples):

            velocity_now = torch.as_tensor(
                simulation.X_input[transition, :, 0:2],
                dtype=dtype,
            )

            velocity_next = torch.as_tensor(
                simulation.Y_target[transition, :, 0:2],
                dtype=dtype,
            )

            pressure_next = torch.as_tensor(
                simulation.Y_target[transition, :, 2],
                dtype=dtype,
            )

            delta_t = float(
                simulation.delta_t[transition]
            )

            if delta_t <= 0.0:
                raise ValueError(
                    f"Simulation {simulation_id}, transition "
                    f"{transition}: non-positive dt={delta_t}."
                )

            continuity_value = continuity_loss(
                velocity_next[:, 0],
                velocity_next[:, 1],
                operators,
                cells,
            ).item()

            momentum_value = momentum_loss(
                velocity_now,
                velocity_next,
                pressure_next,
                delta_t,
                fluid,
                operators,
                cells,
            ).item()

            simulation_continuity.append(
                continuity_value
            )
            simulation_momentum.append(
                momentum_value
            )

            continuity_values.append(
                continuity_value
            )
            momentum_values.append(
                momentum_value
            )

        print(
            f"  Simulation {simulation_id}: "
            f"continuity={np.mean(simulation_continuity):.6e} | "
            f"momentum={np.mean(simulation_momentum):.6e}"
        )

    continuity_reference = float(
        np.mean(continuity_values)
    )

    momentum_reference = float(
        np.mean(momentum_values)
    )

    if (
        not np.isfinite(continuity_reference)
        or continuity_reference <= 0
    ):
        raise ValueError(
            "Invalid continuity reference scale: "
            f"{continuity_reference}."
        )

    if (
        not np.isfinite(momentum_reference)
        or momentum_reference <= 0
    ):
        raise ValueError(
            "Invalid momentum reference scale: "
            f"{momentum_reference}."
        )

    print("\nPhysics reference scales:")
    print(
        f"  continuity_reference = "
        f"{continuity_reference:.6e}"
    )
    print(
        f"  momentum_reference   = "
        f"{momentum_reference:.6e}"
    )

    return continuity_reference, momentum_reference