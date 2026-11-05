"""
The momentum residual on real COMSOL data, term by term. Run it before
turning momentum_weight up.

COMSOL uses its own BDF scheme and finite-element gradients, the
residual a backward Euler step and WLSQ, so the ground-truth residual is
not zero: it is the floor a prediction can reach, and it should be SMALL
compared with the terms that make it up.

    python scripts/test_momentum_residual.py
"""

import torch

from gnn_comsol.data.loading import load_data
from gnn_comsol.physics import (
    FluidProperties,
    build_cell_geometry,
    build_wlsq_operators,
    continuity_residual,
    continuity_loss,
    momentum_residual,
    momentum_loss,
)
from gnn_comsol.physics.navier_stokes import momentum_terms

dataset_path = (
    "C:/Users/giovanni/.comsol/v64/llmatlab/"
    "channel2d_manual_stabilization_200R.mat"
)

# Used only if the .mat does not carry rho and mu, i.e. if the COMSOL
# export in first_database.m did not resolve spf.rho / spf.mu.
FALLBACK_FLUID = FluidProperties(rho=1.0, mu=0.005, source="this script")

# Which transition to look at. Python index, MATLAB index minus one.
TRANSITION = 99


data = load_data(dataset_path, skip_initial=0)

fluid = FluidProperties.from_simulation(data) or FALLBACK_FLUID

operators = build_wlsq_operators(
    data.neighbors,
    data.G_wlsq,
    num_nodes=data.num_nodes,
    dtype=torch.float64,
)

geometry = build_cell_geometry(
    data.pos,
    data.cell_index,
    dtype=torch.float64,
)


def tensor(array):
    return torch.as_tensor(array, dtype=torch.float64)


velocity_now = tensor(data.X_input[TRANSITION, :, 0:2])
velocity_next = tensor(data.Y_target[TRANSITION, :, 0:2])
pressure_next = tensor(data.Y_target[TRANSITION, :, 2])

delta_t = float(data.delta_t[TRANSITION])

interior = ~geometry.boundary_node
interior_cells = interior[geometry.cell_index].all(dim=1)

continuity_losses = []
momentum_losses = []
residual_ratios = []

time_terms = []
convection_terms = []
pressure_terms = []
viscous_terms = []

for transition in range(data.num_samples):

    velocity_now = tensor(
        data.X_input[transition, :, 0:2]
    )

    velocity_next = tensor(
        data.Y_target[transition, :, 0:2]
    )

    pressure_next = tensor(
        data.Y_target[transition, :, 2]
    )

    delta_t = float(
        data.delta_t[transition]
    )

    # Continuity

    cont_loss = continuity_loss(
        velocity_next[:, 0],
        velocity_next[:, 1],
        operators,
        geometry,
    )

    continuity_losses.append(
        cont_loss.item()
    )

    # Momentum terms

    terms = momentum_terms(
        velocity_now,
        velocity_next,
        pressure_next,
        delta_t,
        fluid,
        operators,
        geometry,
    )

    residual = momentum_residual(
        velocity_now,
        velocity_next,
        pressure_next,
        delta_t,
        fluid,
        operators,
        geometry,
    )

    mom_loss = momentum_loss(
        velocity_now,
        velocity_next,
        pressure_next,
        delta_t,
        fluid,
        operators,
        geometry,
    )

    momentum_losses.append(
        mom_loss.item()
    )

    # Interior-cell term magnitudes

    magnitudes = {
        name: value[
            interior_cells
        ].abs().mean().item()
        for name, value in terms.items()
    }

    time_terms.append(
        magnitudes["time"]
    )

    convection_terms.append(
        magnitudes["convection"]
    )

    pressure_terms.append(
        magnitudes["pressure"]
    )

    viscous_terms.append(
        magnitudes["viscous"]
    )

    residual_magnitude = (
        residual[interior_cells]
        .abs()
        .mean()
        .item()
    )

    largest_term = max(
        magnitudes.values()
    )

    residual_ratios.append(
        residual_magnitude / largest_term
    )
print("\n========================================")
print("SETUP")
print("========================================")

print("nodes          :", data.num_nodes)
print("cells          :", geometry.num_cells)
print("interior cells :", int(interior_cells.sum().item()))
print("dt             :", delta_t)
print("fluid          :", fluid)


# Cross-check the fluid against the Reynolds number in the file name

pos = tensor(data.pos)

height = (pos[:, 1].max() - pos[:, 1].min()).item()

speed = tensor(data.X_input[TRANSITION, :, 0:2]).norm(dim=1).max().item()

print("\n========================================")
print("REYNOLDS CROSS-CHECK")
print("========================================")

print(f"channel height (from pos)  : {height:.6g}")
print(f"peak speed  (from the data): {speed:.6g}")
print(
    "implied Re = U H / nu       : "
    f"{fluid.reynolds(speed, height):.6g}"
)
print("compare against the number in the file name.")


# CONTINUITY

continuity = continuity_residual(
    velocity_next[:, 0],
    velocity_next[:, 1],
    operators,
    geometry,
)

print("\n========================================")
print("CONTINUITY RESIDUAL OF THE GROUND TRUTH")
print("========================================")

print("abs mean (all cells)     :", continuity.abs().mean().item())
print(
    "abs mean (interior only) :",
    continuity[interior_cells].abs().mean().item(),
)


# MOMENTUM, TERM BY TERM

terms = momentum_terms(
    velocity_now,
    velocity_next,
    pressure_next,
    delta_t,
    fluid,
    operators,
    geometry,
)

residual = momentum_residual(
    velocity_now,
    velocity_next,
    pressure_next,
    delta_t,
    fluid,
    operators,
    geometry,
)

print("\n========================================")
print("MOMENTUM RESIDUAL OF THE GROUND TRUTH")
print("========================================")

print("\nSize of each term (abs mean over interior cells):\n")

largest = 0.0

for name, value in terms.items():

    magnitude = value[interior_cells].abs().mean().item()

    largest = max(largest, magnitude)

    print(f"  {name:12s} {magnitude:.6e}")

interior_residual = residual[interior_cells].abs().mean().item()

print(f"\n  {'RESIDUAL':12s} {interior_residual:.6e}")

print(
    f"\nresidual / largest term = "
    f"{interior_residual / largest:.4%}"
)

print(
    "\nA few percent means the discretization agrees with COMSOL and\n"
    "the residual is usable as a training signal. Of the same order\n"
    "as the largest term means it is not: check rho and mu first,\n"
    "then whether dt is really the gap between these two snapshots."
)

print("\nBoundary cells, for comparison:")

print(
    "  residual abs mean (all cells) :",
    residual.abs().mean().item(),
)

print(
    "  ratio boundary / interior     :",
    (
        residual[~interior_cells].abs().mean()
        / residual[interior_cells].abs().mean()
    ).item(),
)

print(
    "\nThe boundary cells are expected to be worse: their WLSQ\n"
    "stencils are one-sided. enforce_boundary_values handles them\n"
    "during training by imposing the true velocity there."
)

# MOMENTUM LOSS OF THE COMSOL GROUND TRUTH

loss_momentum = momentum_loss(
    velocity_now,
    velocity_next,
    pressure_next,
    delta_t,
    fluid,
    operators,
    geometry,
)

print("\n========================================")
print("MOMENTUM LOSS OF THE GROUND TRUTH")
print("========================================")

print("Momentum loss =", loss_momentum.item())


import numpy as np


def report(name, values):

    values = np.asarray(values)

    print(f"\n{name}")
    print(f"  mean   : {values.mean():.6e}")
    print(f"  median : {np.median(values):.6e}")
    print(f"  min    : {values.min():.6e}")
    print(f"  max    : {values.max():.6e}")
    print(f"  argmax : {values.argmax()}")


print("\n========================================")
print("GROUND TRUTH PHYSICS - ALL TRANSITIONS")
print("========================================")

report(
    "Continuity loss",
    continuity_losses,
)

report(
    "Momentum loss",
    momentum_losses,
)

report(
    "Time term",
    time_terms,
)

report(
    "Convection term",
    convection_terms,
)

report(
    "Pressure term",
    pressure_terms,
)

report(
    "Viscous term",
    viscous_terms,
)

ratios = np.asarray(
    residual_ratios
)

print("\nResidual / largest momentum term")

print(
    f"  mean   : "
    f"{100 * ratios.mean():.3f}%"
)

print(
    f"  median : "
    f"{100 * np.median(ratios):.3f}%"
)

print(
    f"  min    : "
    f"{100 * ratios.min():.3f}%"
)

print(
    f"  max    : "
    f"{100 * ratios.max():.3f}%"
)

print(
    f"  worst transition: "
    f"{ratios.argmax()}"
)
