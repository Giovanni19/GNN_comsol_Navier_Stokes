"""
Continuity and momentum loss of the COMSOL ground truth over every
transition: the discretization floor across the whole simulation.
"""

import numpy as np
import torch

from gnn_comsol.data.loading import load_data
from gnn_comsol.physics import (
    FluidProperties,
    build_cell_geometry,
    build_wlsq_operators,
    continuity_loss,
    momentum_loss,
    momentum_residual,
)

from gnn_comsol.physics.navier_stokes import momentum_terms


# CONFIGURATION

dataset_path = (
    "C:/Users/giovanni/.comsol/v64/llmatlab/"
    "channel2d_manual_stabilization_200R.mat"
)

FALLBACK_FLUID = FluidProperties(
    rho=1.0,
    mu=0.005,
    source="this script",
)


# LOAD DATASET

data = load_data(
    dataset_path,
    skip_initial=0,
)

fluid = (
    FluidProperties.from_simulation(data)
    or FALLBACK_FLUID
)


# BUILD GEOMETRY ONCE

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
    return torch.as_tensor(
        array,
        dtype=torch.float64,
    )


# INTERIOR CELLS

interior_nodes = ~geometry.boundary_node

interior_cells = interior_nodes[
    geometry.cell_index
].all(dim=1)


# STORAGE

num_transitions = len(data.delta_t)

continuity_losses = np.zeros(num_transitions)

momentum_losses = np.zeros(num_transitions)

momentum_relative = np.zeros(num_transitions)

momentum_abs_mean = np.zeros(num_transitions)

time_term_mean = np.zeros(num_transitions)
convection_term_mean = np.zeros(num_transitions)
pressure_term_mean = np.zeros(num_transitions)
viscous_term_mean = np.zeros(num_transitions)


print("\n========================================")
print("PHYSICS RESIDUAL ANALYSIS")
print("========================================")

print("Dataset        :", dataset_path)
print("Nodes          :", data.num_nodes)
print("Cells          :", geometry.num_cells)
print(
    "Interior cells :",
    int(interior_cells.sum().item()),
)
print("Transitions    :", num_transitions)
print("Fluid          :", fluid)


# LOOP OVER ALL TRANSITIONS

with torch.no_grad():

    for k in range(num_transitions):

        velocity_now = tensor(
            data.X_input[k, :, 0:2]
        )

        velocity_next = tensor(
            data.Y_target[k, :, 0:2]
        )

        pressure_next = tensor(
            data.Y_target[k, :, 2]
        )

        delta_t = float(
            data.delta_t[k]
        )

        # CONTINUITY LOSS

        continuity_losses[k] = continuity_loss(
            velocity_next[:, 0],
            velocity_next[:, 1],
            operators,
            geometry,
        ).item()

        # MOMENTUM LOSS

        momentum_losses[k] = momentum_loss(
            velocity_now,
            velocity_next,
            pressure_next,
            delta_t,
            fluid,
            operators,
            geometry,
        ).item()

        # MOMENTUM TERMS

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

        # INTERIOR-CELL STATISTICS

        term_means = {}

        for name, value in terms.items():

            term_means[name] = (
                value[interior_cells]
                .abs()
                .mean()
                .item()
            )

        # These names must match momentum_terms().
        time_term_mean[k] = term_means["time"]
        convection_term_mean[k] = term_means["convection"]
        pressure_term_mean[k] = term_means["pressure"]
        viscous_term_mean[k] = term_means["viscous"]

        residual_mean = (
            residual[interior_cells]
            .abs()
            .mean()
            .item()
        )

        momentum_abs_mean[k] = residual_mean

        largest_term = max(
            term_means.values()
        )

        momentum_relative[k] = (
            residual_mean / largest_term
            if largest_term > 0
            else np.nan
        )

        # Progress
        if (
            k == 0
            or (k + 1) % 25 == 0
            or k == num_transitions - 1
        ):

            print(
                f"{k + 1:4d}/{num_transitions} | "
                f"dt={delta_t:.4e} | "
                f"Lcont={continuity_losses[k]:.4e} | "
                f"Lmom={momentum_losses[k]:.4e} | "
                f"Rrel={100 * momentum_relative[k]:.2f}%"
            )


# SUMMARY

def print_stats(name, values):

    finite = values[np.isfinite(values)]

    print(f"\n{name}")

    print(
        f"  min    = {np.min(finite):.6e}"
    )

    print(
        f"  median = {np.median(finite):.6e}"
    )

    print(
        f"  mean   = {np.mean(finite):.6e}"
    )

    print(
        f"  max    = {np.max(finite):.6e}"
    )


print("\n========================================")
print("GROUND-TRUTH PHYSICS FLOOR")
print("========================================")

print_stats(
    "Continuity loss",
    continuity_losses,
)

print_stats(
    "Momentum loss",
    momentum_losses,
)

print_stats(
    "Momentum residual / largest term",
    momentum_relative,
)


# WORST TRANSITIONS

worst = np.argsort(
    momentum_relative
)[::-1][:10]


print("\n========================================")
print("10 WORST MOMENTUM TRANSITIONS")
print("========================================")

for k in worst:

    print(
        f"k={k:4d} | "
        f"dt={float(data.delta_t[k]):.4e} | "
        f"Lmom={momentum_losses[k]:.4e} | "
        f"Rrel={100 * momentum_relative[k]:.2f}%"
    )


# SAVE RESULTS

output_file = (
    "physics_residual_analysis.npz"
)

np.savez(
    output_file,

    continuity_loss=continuity_losses,
    momentum_loss=momentum_losses,
    momentum_relative=momentum_relative,
    momentum_abs_mean=momentum_abs_mean,

    time_term=time_term_mean,
    convection_term=convection_term_mean,
    pressure_term=pressure_term_mean,
    viscous_term=viscous_term_mean,

    delta_t=np.asarray(data.delta_t),
)

print("\nResults saved to:")
print(output_file)
