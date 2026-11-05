"""
Train / validation / test splitting.

Modes:
    temporal   split samples into contiguous time blocks
    random     split shuffled samples
    simulation split complete simulations
"""

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class SimulationSplits:
    train: list
    val: list
    test: list


def compute_split_indices(
    n_samples,
    mode="temporal",
    train_fraction=0.70,
    val_fraction=0.15,
    gap=1,
    seed=68,
):
    """Return train, validation and test sample indices."""

    n_train = int(train_fraction * n_samples)
    n_val = int(val_fraction * n_samples)

    if mode == "temporal":
        indices = np.arange(n_samples)

        train = indices[:n_train]

        val_start = n_train + gap
        val = indices[val_start:val_start + n_val]

        test_start = val_start + n_val + gap
        test = indices[test_start:]

        if not len(train) or not len(val) or not len(test):
            raise ValueError(
                "Temporal split leaves an empty block: "
                f"n_samples={n_samples}, "
                f"train_fraction={train_fraction}, "
                f"val_fraction={val_fraction}, gap={gap}."
            )

        return train, val, test

    if mode == "random":
        indices = np.random.default_rng(seed).permutation(n_samples)

        return (
            indices[:n_train],
            indices[n_train:n_train + n_val],
            indices[n_train + n_val:],
        )

    raise ValueError(
        f"Unknown split mode {mode!r}. "
        "Expected 'temporal' or 'random'."
    )


def split_simulations(
    simulations,
    train_ids,
    val_ids,
    test_ids,
):
    """Split complete simulations using explicit simulation IDs."""

    if not simulations:
        raise ValueError(
            "At least one simulation is required."
        )

    train_ids = list(train_ids)
    val_ids = list(val_ids)
    test_ids = list(test_ids)

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    # Check that the three splits do not overlap.
    if train_set & val_set:
        raise ValueError(
            "Train and validation simulation IDs overlap."
        )

    if train_set & test_set:
        raise ValueError(
            "Train and test simulation IDs overlap."
        )

    if val_set & test_set:
        raise ValueError(
            "Validation and test simulation IDs overlap."
        )

    # Map simulation IDs to simulation objects.
    simulation_by_id = {
        simulation.simulation_id: simulation
        for simulation in simulations
    }

    available_ids = set(simulation_by_id)

    requested_ids = train_set | val_set | test_set

    # Check that all requested IDs exist.
    unknown_ids = requested_ids - available_ids

    if unknown_ids:
        raise ValueError(
            f"Unknown simulation IDs: {sorted(unknown_ids)}. "
            f"Available IDs: {sorted(available_ids)}."
        )

    # Check that every loaded simulation is assigned.
    unassigned_ids = available_ids - requested_ids

    if unassigned_ids:
        raise ValueError(
            "Some simulations were not assigned to any split: "
            f"{sorted(unassigned_ids)}."
        )

    # Training cannot be empty.
    if not train_ids:
        raise ValueError(
            "At least one training simulation is required."
        )

    return SimulationSplits(
        train=[
            simulation_by_id[i]
            for i in train_ids
        ],
        val=[
            simulation_by_id[i]
            for i in val_ids
        ],
        test=[
            simulation_by_id[i]
            for i in test_ids
        ],
    )


def _subset_optional(values, indices):
    """Subset a sample-dependent optional array."""
    return None if values is None else values[indices]


def subset_simulation(simulation, indices):
    """Restrict one simulation to selected transitions."""

    indices = np.asarray(indices, dtype=np.int64)

    return replace(
        simulation,
        X_input=simulation.X_input[indices],
        Y_target=simulation.Y_target[indices],
        delta_t=simulation.delta_t[indices],

        physics_features=_subset_optional(
            simulation.physics_features,
            indices,
        ),

        velocity_bc_target=_subset_optional(
            simulation.velocity_bc_target,
            indices,
        ),

        pressure_bc_target=_subset_optional(
            simulation.pressure_bc_target,
            indices,
        ),

        jacobian_values=_subset_optional(
            simulation.jacobian_values,
            indices,
        ),

        jacobian_target_time=_subset_optional(
            simulation.jacobian_target_time,
            indices,
        ),
    )


def split_simulations_by_sample(
    simulations,
    mode="temporal",
    train_fraction=0.70,
    val_fraction=0.15,
    gap=1,
    seed=68,
):
    """
    Split each simulation independently along its transitions.

    Every simulation contributes to train, validation and test, so this
    tests temporal generalization rather than unseen-geometry
    generalization.
    """

    if mode not in ("temporal", "random"):
        raise ValueError(
            f"Unsupported sample split mode {mode!r}."
        )

    if not simulations:
        raise ValueError(
            "At least one simulation is required."
        )

    splits = SimulationSplits([], [], [])

    for simulation in simulations:

        if len(simulation.delta_t) != simulation.num_samples:
            raise ValueError(
                f"Simulation {simulation.simulation_id}: "
                "delta_t and samples are misaligned."
            )

        indices = compute_split_indices(
            simulation.num_samples,
            mode=mode,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            gap=gap,
            seed=seed,
        )

        for block, block_indices in zip(
            (splits.train, splits.val, splits.test),
            indices,
        ):
            block.append(
                subset_simulation(
                    simulation,
                    block_indices,
                )
            )

    return splits
