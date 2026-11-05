import numpy as np
import torch
from torch_geometric.data import Data

from gnn_comsol import data as gdata
from gnn_comsol.boundary import (
    apply_velocity_dirichlet,
    apply_pressure_dirichlet,
    boundary_consistency,
    velocity_dirichlet_violation,
)


def evaluate(
    evaluation,
    velocity,
    pressure,
    hierarchies,
    device,
    hard_bc=True,
):
    """
    One-step prediction for every sample of `evaluation`, in physical
    units.
    hard_bc
        Hard-impose all available Dirichlet boundary conditions on the
        reconstructed physical prediction: velocity at wall/inlet nodes
        and pressure at pressure-Dirichlet nodes. The corrected fields
        are used for all reported errors and are written to the MATLAB
        prediction file.

    Returns (rmse, pressure_absolute_error, velocity_absolute_error,
    pressure_previous_target_error, pressure_previous_prediction_error,
    velocity_previous_target_error, velocity_previous_prediction_error,
    predictions, boundary), the last one being the diagnostics of the
    replacement, or None when it did not run.
    """
    if evaluation.U_inf is None:
        raise ValueError("Evaluation dataset has no U_inf.")

    if evaluation.rho is None:
        raise ValueError("Evaluation dataset has no rho.")
    if evaluation.R is None:
        raise ValueError("Evaluation dataset has no cylinder radius R.")

    U_inf = float(evaluation.U_inf)
    rho = float(evaluation.rho)

    state_scale = np.array(
        [
            U_inf,
            U_inf,
            rho * U_inf**2,
        ],
        dtype=np.float64,
    )
    X_norm = evaluation.X_input / state_scale

    R = float(evaluation.R)

    if R <= 0:
        raise ValueError(f"R must be positive, got {R}.")

    D = 2.0 * R

    dt_norm = evaluation.delta_t * U_inf / D

    state_scale_torch = torch.as_tensor(
        state_scale,
        dtype=torch.float32,
        device=device,
    )

    # Physics-derived features, scaled with each network's own normalizer

    velocity_uses_physics = velocity.metadata.get(
        "use_physics_features",
        False,
    )

    pressure_uses_physics = pressure.metadata.get(
        "use_physics_features",
        False,
    )

    velocity_physics_norm = None
    pressure_physics_norm = None

    if velocity_uses_physics:

        if evaluation.physics_features is None:
            raise ValueError(
                "The velocity network was trained on physics-derived "
                "features, but this dataset does not contain any. It "
                "expects "
                f"{velocity.metadata.get('physics_feature_names')}."
            )

        if velocity.physics_normalizer is None:
            raise ValueError(
                "The velocity checkpoint says it uses physics features "
                "but does not carry their scaling, so they cannot be "
                "reproduced. Retrain with the current code."
            )

        velocity_physics_norm = (
            velocity.physics_normalizer.transform(
                evaluation.physics_features
            )
        )

    if pressure_uses_physics:

        if evaluation.physics_features is None:
            raise ValueError(
                "The pressure network was trained on physics-derived "
                "features, but this dataset does not contain any. It "
                "expects "
                f"{pressure.metadata.get('physics_feature_names')}."
            )

        if pressure.physics_normalizer is None:
            raise ValueError(
                "The pressure checkpoint says it uses physics features "
                "but does not carry their scaling, so they cannot be "
                "reproduced. Retrain with the current code."
            )

        pressure_physics_norm = (
            pressure.physics_normalizer.transform(
                evaluation.physics_features
            )
        )
    # Geometry features: static per node, broadcast over timesteps

    velocity_uses_geometry = velocity.metadata.get(
        "use_geometry_features",
        False,
    )

    pressure_uses_geometry = pressure.metadata.get(
        "use_geometry_features",
        False,
    )

    velocity_geometry_norm = None
    pressure_geometry_norm = None

    if velocity_uses_geometry:

        if evaluation.geometry_features is None:
            raise ValueError(
                "The velocity network requires geometry features, "
                "but the dataset does not contain them."
            )

        if velocity.geometry_normalizer is None:
            raise ValueError(
                "The velocity checkpoint requires geometry features "
                "but does not contain their normalizer."
            )

        velocity_geometry_norm = (
            velocity.geometry_normalizer.transform(
                evaluation.geometry_features
            )
        )

    if pressure_uses_geometry:

        if evaluation.geometry_features is None:
            raise ValueError(
                "The pressure network requires geometry features, "
                "but the dataset does not contain them."
            )

        if pressure.geometry_normalizer is None:
            raise ValueError(
                "The pressure checkpoint requires geometry features "
                "but does not contain their normalizer."
            )

        pressure_geometry_norm = (
            pressure.geometry_normalizer.transform(
                evaluation.geometry_features
            )
        )

    node_type_features = None

    if (
        velocity.metadata.get("use_node_type_features", False)
        or pressure.metadata.get("use_node_type_features", False)
    ):
        if (
            evaluation.wall_mask is None
            or evaluation.inlet_mask is None
            or evaluation.outlet_mask is None
        ):
            raise ValueError(
                "Checkpoint requires node-type features, but the "
                "evaluation dataset does not contain wall_mask, "
                "inlet_mask and outlet_mask."
            )

        node_type_features = gdata.build_node_type_features(
            evaluation.wall_mask,
            evaluation.inlet_mask,
            evaluation.outlet_mask,
        )
    velocity_features = gdata.build_features(
        velocity.metadata["features"],
        X_norm,
        dt_norm,

        physics_features=(
            velocity_physics_norm
            if velocity_uses_physics
            else None
        ),

        geometry_features=(
            velocity_geometry_norm
            if velocity_uses_geometry
            else None
        ),

        node_type_features=(
            node_type_features
            if velocity.metadata.get(
                "use_node_type_features",
                False,
            )
            else None
        ),
    )

    # Safety check
    expected_velocity_features = velocity.metadata["num_in"]

    if velocity_features.shape[-1] != expected_velocity_features:
        raise ValueError(
            f"Velocity model expects {expected_velocity_features} features, "
            f"but evaluation built {velocity_features.shape[-1]}."
        )

    # Delta prediction: the network's raw output is a nondimensional
    # INCREMENT, not the absolute next state. It is multiplied by the
    # physical state scale and added to the physical input state.

    velocity_predicts_delta = velocity.metadata.get(
        "predict_delta", False
    )

    pressure_predicts_delta = pressure.metadata.get(
        "predict_delta", False
    )
    for name, bundle, predicts_delta in (
        ("velocity", velocity, velocity_predicts_delta),
        ("pressure", pressure, pressure_predicts_delta),
    ):
        if predicts_delta:
            delta_scaling = bundle.metadata.get("delta_scaling")

            if delta_scaling != "physical_nondimensionalization":
                raise ValueError(
                    f"{name} predicts delta, but its checkpoint has "
                    f"delta_scaling={delta_scaling!r}. "
                    "This evaluator expects "
                    "'physical_nondimensionalization'."
                )

    topology = gdata.build_graph_topologies(
        [evaluation]
    )[evaluation.simulation_id]

    edge_index = topology["edge_index"].to(device)
    edge_weight = topology["edge_weight"].to(device)
    edge_attr = topology["edge_attr"].to(device)
    pos = topology["pos"].to(device)
    # ============================================================
    # HARD DIRICHLET BOUNDARY CONDITIONS
    # ============================================================

    has_velocity_boundary = (
        evaluation.velocity_dirichlet_mask is not None
        and evaluation.velocity_bc_target is not None
    )

    has_pressure_boundary = (
        evaluation.pressure_dirichlet_mask is not None
        and evaluation.pressure_bc_target is not None
    )

    enforce_velocity_boundary = (
        hard_bc and has_velocity_boundary
    )

    enforce_pressure_boundary = (
        hard_bc and has_pressure_boundary
    )

    if hard_bc and not has_velocity_boundary:
        print(
            "\nNOTE: this dataset carries no velocity Dirichlet "
            "boundary conditions. Velocity is left as predicted."
        )

    if hard_bc and not has_pressure_boundary:
        print(
            "\nNOTE: this dataset carries no pressure Dirichlet "
            "boundary conditions. Pressure is left as predicted."
        )

    velocity_dirichlet_mask = None
    velocity_bc = None

    pressure_dirichlet_mask = None
    pressure_bc = None

    if enforce_velocity_boundary:

        velocity_dirichlet_mask = torch.as_tensor(
            evaluation.velocity_dirichlet_mask,
            dtype=torch.bool,
            device=device,
        )

        velocity_bc = torch.as_tensor(
            evaluation.velocity_bc_target,
            dtype=torch.float32,
            device=device,
        )

        check_boundary_against_target(evaluation)

    if enforce_pressure_boundary:

        pressure_dirichlet_mask = torch.as_tensor(
            evaluation.pressure_dirichlet_mask,
            dtype=torch.bool,
            device=device,
        )

        pressure_bc = torch.as_tensor(
            evaluation.pressure_bc_target,
            dtype=torch.float32,
            device=device,
        )

    boundary_violation = {
        "rms": np.zeros(evaluation.num_samples),
        "max": np.zeros(evaluation.num_samples),
    }

    num_samples = evaluation.num_samples

    predictions = np.zeros(
        (num_samples, evaluation.num_nodes, 3),
        dtype=np.float32
    )

    rmse = {
        name: np.zeros(num_samples)
        for name in gdata.VARIABLE_NAMES
    }

    pressure_absolute_error = np.zeros(
        (num_samples, evaluation.num_nodes),
        dtype=np.float32
    )

    velocity_absolute_error = np.zeros(
        (num_samples, evaluation.num_nodes),
        dtype=np.float32
    )

    # Three-way diagnostics for the Newton initial guess:
    #   prediction vs target, previous state vs target, previous state vs prediction.
    pressure_previous_target_error = np.zeros_like(pressure_absolute_error)
    pressure_previous_prediction_error = np.zeros_like(pressure_absolute_error)
    velocity_previous_target_error = np.zeros_like(velocity_absolute_error)
    velocity_previous_prediction_error = np.zeros_like(velocity_absolute_error)
    velocity_hierarchy = hierarchies.get("velocity")
    pressure_hierarchy = hierarchies.get("pressure")
    if (
        velocity.metadata["architecture"] == "bsms"
        and velocity_hierarchy is None
    ):
        raise ValueError(
            "Velocity uses BSMS but no velocity BSMS hierarchy was provided."
        )

    if (
        pressure.metadata["architecture"] == "bsms"
        and pressure_hierarchy is None
    ):
        raise ValueError(
            "Pressure uses BSMS but no pressure BSMS hierarchy was provided."
        )
    print(f"\nEvaluating {num_samples} held-out samples...")

    with torch.no_grad():

        for timestep in range(num_samples):

            # Velocity

            velocity_graph = Data(
                x=torch.as_tensor(
                    velocity_features[timestep],
                    dtype=torch.float32,
                    device=device,
                ),
                edge_index=edge_index,
                edge_weight=edge_weight,
                edge_attr=edge_attr,
                pos=pos,
            )

            if velocity.metadata["architecture"] == "bsms":

                velocity_pred_norm = velocity.model(
                    velocity_graph,
                    velocity_hierarchy["edge_indices"],
                    velocity_hierarchy["pool_indices"],
                )

            else:

                velocity_pred_norm = velocity.model(
                    velocity_graph
                )
            # Pressure, with the same feature builder as training
            pressure_features = gdata.build_features(
                pressure.metadata["features"],
                X_norm[timestep:timestep + 1],
                dt_norm[timestep:timestep + 1],
                physics_features=(
                    pressure_physics_norm[timestep:timestep + 1]
                    if pressure_physics_norm is not None
                    else None
                ),
                geometry_features=(
                    pressure_geometry_norm
                    if pressure_uses_geometry
                    else None
                ),
                node_type_features=(
                    node_type_features
                    if pressure.metadata.get(
                        "use_node_type_features",
                        False,
                    )
                    else None
                ),
            )[0]

            expected = pressure.metadata["num_in"]

            if pressure_features.shape[-1] != expected:
                raise ValueError(
                    f"The pressure model expects {expected} input "
                    f"features, but {pressure_features.shape[-1]} were "
                    "assembled."
                )

            pressure_graph = Data(
                x=torch.as_tensor(
                    pressure_features,
                    dtype=torch.float32,
                    device=device,
                ),
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=pos,
            )

            if pressure.metadata["architecture"] == "bsms":

                pressure_pred_norm = pressure.model(
                    pressure_graph,
                    pressure_hierarchy["edge_indices"],
                    pressure_hierarchy["pool_indices"],
                )

            else:

                pressure_pred_norm = pressure.model(
                    pressure_graph
                )
            # Back to physical units, velocity and pressure separately
            # (either may predict an increment); always absolute fields.

            X_prev_physical = torch.as_tensor(
                evaluation.X_input[timestep],
                dtype=torch.float32,
                device=device
            )
            velocity_scale = state_scale_torch[gdata.VELOCITY_COLUMNS]

            if velocity_predicts_delta:

                velocity_pred_phys = (
                    X_prev_physical[:, gdata.VELOCITY_COLUMNS]
                    + velocity_pred_norm * velocity_scale
                )

            else:

                velocity_pred_phys = (
                    velocity_pred_norm * velocity_scale
                )

            # Hard velocity BCs, measuring the violation first

            if enforce_velocity_boundary:

                violation = velocity_dirichlet_violation(
                    velocity_pred_phys,
                    velocity_dirichlet_mask,
                    velocity_bc[timestep],
                )

                boundary_violation["rms"][timestep] = violation["rms"]
                boundary_violation["max"][timestep] = violation["max"]

                velocity_pred_phys = apply_velocity_dirichlet(
                    velocity_pred_phys,
                    velocity_dirichlet_mask,
                    velocity_bc[timestep],
                )

            pressure_scale = state_scale_torch[gdata.PRESSURE_COLUMNS]

            if pressure_predicts_delta:

                pressure_pred_phys = (
                    X_prev_physical[:, gdata.PRESSURE_COLUMNS]
                    + pressure_pred_norm * pressure_scale
                )

            else:

                pressure_pred_phys = (
                    pressure_pred_norm * pressure_scale
                )
            # Hard pressure Dirichlet BC

            if enforce_pressure_boundary:

                pressure_pred_phys_1d = (
                    pressure_pred_phys.squeeze(-1)
                )

                pressure_pred_phys_1d = apply_pressure_dirichlet(
                    pressure_pred_phys_1d,
                    pressure_dirichlet_mask,
                    pressure_bc[timestep],
                )

                pressure_pred_phys = (
                    pressure_pred_phys_1d.unsqueeze(-1)
                )
            Y_pred = torch.cat(
                [velocity_pred_phys, pressure_pred_phys],
                dim=-1
            ).cpu().numpy()

            predictions[timestep] = Y_pred

            error = Y_pred - evaluation.Y_target[timestep]

            pressure_absolute_error[timestep] = np.abs(error[:, 2])

            velocity_absolute_error[timestep] = np.sqrt(
                error[:, 0] ** 2 + error[:, 1] ** 2
            )

            previous = evaluation.X_input[timestep]
            target = evaluation.Y_target[timestep]

            pressure_previous_target_error[timestep] = np.abs(
                previous[:, 2] - target[:, 2]
            )
            pressure_previous_prediction_error[timestep] = np.abs(
                previous[:, 2] - Y_pred[:, 2]
            )

            velocity_previous_target_error[timestep] = np.sqrt(
                (previous[:, 0] - target[:, 0]) ** 2
                + (previous[:, 1] - target[:, 1]) ** 2
            )
            velocity_previous_prediction_error[timestep] = np.sqrt(
                (previous[:, 0] - Y_pred[:, 0]) ** 2
                + (previous[:, 1] - Y_pred[:, 1]) ** 2
            )

            for column, name in enumerate(gdata.VARIABLE_NAMES):
                rmse[name][timestep] = np.sqrt(
                    np.mean(error[:, column] ** 2)
                )

            if timestep % 100 == 0 or timestep == num_samples - 1:
                print(
                    f"  {timestep + 1}/{num_samples} | "
                    + " | ".join(
                        f"RMSE {name}={rmse[name][timestep]:.3e}"
                        for name in gdata.VARIABLE_NAMES
                    )
                )

    boundary = None

    if enforce_velocity_boundary:

        boundary = {
            "num_nodes": int(
                velocity_dirichlet_mask.sum()
            ),
            "num_total": evaluation.num_nodes,
            "rms": boundary_violation["rms"],
            "max": boundary_violation["max"],
        }

    return (
        rmse,
        pressure_absolute_error,
        velocity_absolute_error,
        pressure_previous_target_error,
        pressure_previous_prediction_error,
        velocity_previous_target_error,
        velocity_previous_prediction_error,
        predictions,
        boundary
    )

def check_boundary_against_target(evaluation, tolerance=1e-6):
    """
    Report whether the prescribed boundary condition matches the COMSOL
    solution at the Dirichlet nodes (see boundary.boundary_consistency).
    Warns rather than raises.
    """

    consistency = boundary_consistency(
        evaluation.velocity_bc_target,
        evaluation.velocity_dirichlet_mask,
        evaluation.Y_target[:, :, gdata.VELOCITY_COLUMNS],
    )

    print(
        f"\nBoundary condition vs COMSOL solution at the "
        f"{consistency['num_nodes']} Dirichlet nodes: "
        f"largest disagreement {consistency['largest']:.3e} "
        f"({100 * consistency['relative']:.3f}% of max |velocity|)"
    )

    if consistency["largest"] > tolerance * max(consistency["scale"], 1.0):
        print(
            "  WARNING: the prescribed boundary condition and the "
            "solved field disagree by more than the export tolerance. "
            "Check the wall/inlet/outlet boundary ids in "
            "first_database.m before trusting the imposed values - in "
            "particular the corner nodes, which belong to a wall and "
            "to the inlet at the same time."
        )