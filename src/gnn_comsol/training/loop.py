"""
Coupled velocity-pressure training.

This module controls the complete coupled training process:
setup, optimization loop, best-state restoration, checkpoint saving,
and training-history plotting.
"""

import copy
import time
import numpy as np
import torch
from torch_geometric.data import Data, Batch

from .output import save_coupled_training_outputs
from gnn_comsol.boundary import (
    apply_velocity_dirichlet_to_delta,
    apply_pressure_dirichlet_to_delta,
)
from gnn_comsol.data.normalization import (
    PRESSURE_COLUMNS,
    STATE_COLUMNS,
    VELOCITY_COLUMNS,
)

from .losses import (
    fem_linearized_residual_metrics,
    mse_topk_loss,
)
from .setup import setup_coupled_training


def train_coupled_loop(
    velocity_net,
    pressure_net,
    train_loaders,
    val_loaders,
    graph_topologies,
    bsms_hierarchies,
    physics,
    velocity_optimizer,
    pressure_optimizer,
    criterion,
    num_epochs,
    device,
    velocity_architecture,
    pressure_architecture,
    velocity_scheduler=None,
    pressure_scheduler=None,
    velocity_data_weight=1.0,
    pressure_data_weight=1.0,
    data_loss_type="mse",
    topk_fraction=0.10,
    topk_weight=1.0,
    physics_weight_strategy="fixed",
    continuity_weight=0.01,
    momentum_weight=0.01,
    newton_weight=0.0,
    newton_loss_type="mse",
    newton_topk_fraction=0.10,
    newton_topk_weight=1.0,
    dynamic_weight_learning_rate=1e-4,
    predict_velocity_delta=False,
    predict_pressure_delta=False,
    verbose=True,
    epoch_callback=None,
):
    """
    Joint training of separate velocity and pressure networks.

    Velocity and pressure keep separate architectures and optimizers,
    but are coupled through a common Navier-Stokes physics loss.

    Total loss:

        L =
            lambda_u * L_velocity_data
          + lambda_p * L_pressure_data
          + lambda_c * L_continuity / continuity_reference
          + lambda_m * L_momentum / momentum_reference
          + newton_weight * L_newton

    The data terms are MSE or MSE + top-k (data_loss_type). The two
    residuals are divided by their COMSOL ground-truth floor
    (CoupledPhysicsLoss), and with physics_weight_strategy="dynamic"
    lambda_c and lambda_m are raised by gradient ascent at every batch.


    The momentum residual uses BOTH predicted velocity and predicted
    pressure, so its gradient updates both networks.

    Without validation loaders (split mode all_train) the weights of
    the last epoch are returned instead of the best ones.
    """
    if newton_weight < 0.0:
        raise ValueError(
            "newton_weight must be non-negative."
        )

    # The other Newton settings are unused, and unchecked, when it is off.
    if newton_weight > 0.0:

        if newton_loss_type not in {"mse", "mse_topk"}:
            raise ValueError(
                "newton_loss_type must be 'mse' or 'mse_topk', "
                f"got {newton_loss_type!r}."
            )

        if not 0.0 < newton_topk_fraction <= 1.0:
            raise ValueError(
                "newton_topk_fraction must satisfy "
                "0 < newton_topk_fraction <= 1."
            )

        if newton_topk_weight < 0.0:
            raise ValueError(
                "newton_topk_weight must be non-negative."
            )
    if data_loss_type not in {"mse", "mse_topk"}:
        raise ValueError(
            "data_loss_type must be 'mse' or 'mse_topk', "
            f"got {data_loss_type!r}."
        )
    if physics_weight_strategy not in {"fixed", "dynamic"}:
        raise ValueError(
            "physics_weight_strategy must be 'fixed' or 'dynamic', "
            f"got {physics_weight_strategy!r}."
        )

    if continuity_weight < 0.0:
        raise ValueError(
            "continuity_weight must be non-negative."
        )

    if momentum_weight < 0.0:
        raise ValueError(
            "momentum_weight must be non-negative."
        )

    if (
        physics_weight_strategy == "dynamic"
        and dynamic_weight_learning_rate <= 0.0
    ):
        raise ValueError(
            "dynamic_weight_learning_rate must be positive "
            "when physics_weight_strategy='dynamic'."
        )
    if physics_weight_strategy == "dynamic":

        continuity_weight_param = torch.nn.Parameter(
            torch.tensor(
                float(continuity_weight),
                dtype=torch.float32,
                device=device,
            )
        )

        momentum_weight_param = torch.nn.Parameter(
            torch.tensor(
                float(momentum_weight),
                dtype=torch.float32,
                device=device,
            )
        )

        weight_optimizer = torch.optim.Adam(
            [
                continuity_weight_param,
                momentum_weight_param,
            ],
            lr=dynamic_weight_learning_rate,
        )

    else:

        continuity_weight_param = torch.tensor(
            float(continuity_weight),
            dtype=torch.float32,
            device=device,
        )

        momentum_weight_param = torch.tensor(
            float(momentum_weight),
            dtype=torch.float32,
            device=device,
        )

        weight_optimizer = None
    best_val_loss = np.inf

    best_velocity_state = None
    best_pressure_state = None

    train_loss_history = []
    val_loss_history = []
    has_validation = bool(val_loaders)

    if verbose and not has_validation:
        print(
            "\nNo validation set: all available data are used for training. "
            "The final epoch will be kept."
        )
    # One epoch over either train or validation loaders

    def run_epoch(loaders, training):

        if training:
            velocity_net.train()
            pressure_net.train()
        else:
            velocity_net.eval()
            pressure_net.eval()

        total_loss_sum = 0.0
        sample_count = 0

        term_sums = {
            "velocity_data": 0.0,
            "pressure_data": 0.0,
            "velocity_mse": 0.0,
            "velocity_topk": 0.0,

            "pressure_mse": 0.0,
            "pressure_topk": 0.0,

            # Physical residuals
            "continuity": 0.0,
            "momentum": 0.0,

            # Residuals normalized by the COMSOL training-set floor
            "continuity_normalized": 0.0,
            "momentum_normalized": 0.0,

            # Physics weights
            "continuity_weight": 0.0,
            "momentum_weight": 0.0,

            # Actual weighted contribution to the objective
            "continuity_weighted": 0.0,
            "momentum_weighted": 0.0,

            #Newton weight
            "newton": 0.0,
            "newton_mse": 0.0,
            "newton_topk": 0.0,
            "newton_weighted": 0.0,

        }

        simulation_ids = list(loaders.keys())

        if training:
            np.random.shuffle(simulation_ids)

        context = (
            torch.enable_grad()
            if training
            else torch.no_grad()
        )

        with context:

            for simulation_id in simulation_ids:

                loader = loaders[simulation_id]

                # Static mesh/FEM topology
                topology = graph_topologies[simulation_id]

                edge_index = topology["edge_index"].to(device)
                edge_weight = topology["edge_weight"].to(device)
                edge_attr = topology["edge_attr"].to(device)
                pos = topology["pos"].to(device)
                # FEM Jacobian metadata, only for the Newton loss

                if newton_weight > 0.0:

                    jacobian_row = topology["jacobian_row"]
                    jacobian_col = topology["jacobian_col"]
                    jacobian_shape = topology["jacobian_shape"]

                    jacobian_dof_p = topology["jacobian_dof_p"]
                    jacobian_dof_u = topology["jacobian_dof_u"]
                    jacobian_dof_v = topology["jacobian_dof_v"]

                    if jacobian_row is None or jacobian_col is None:
                        raise ValueError(
                            f"Simulation {simulation_id} has no FEM Jacobian pattern."
                        )

                    if jacobian_shape is None:
                        raise ValueError(
                            f"Simulation {simulation_id} has no FEM Jacobian shape."
                        )

                    if (
                        jacobian_dof_p is None
                        or jacobian_dof_u is None
                        or jacobian_dof_v is None
                    ):
                        raise ValueError(
                            f"Simulation {simulation_id} has no FEM DOF mapping."
                        )

                    jacobian_row = jacobian_row.to(device)
                    jacobian_col = jacobian_col.to(device)

                    jacobian_dof_p = jacobian_dof_p.to(device)
                    jacobian_dof_u = jacobian_dof_u.to(device)
                    jacobian_dof_v = jacobian_dof_v.to(device)

                # Static BSMS hierarchies.
                # Velocity and pressure may independently use BSMS.

                velocity_ms_edges = None
                velocity_ms_ids = None

                if velocity_architecture == "bsms":

                    hierarchy = bsms_hierarchies["velocity"][simulation_id]

                    velocity_ms_edges = [
                        edge.to(device)
                        for edge in hierarchy["edge_indices"]
                    ]

                    velocity_ms_ids = [
                        indices.to(device)
                        for indices in hierarchy["pool_indices"]
                    ]


                pressure_ms_edges = None
                pressure_ms_ids = None

                if pressure_architecture == "bsms":

                    hierarchy = bsms_hierarchies["pressure"][simulation_id]

                    pressure_ms_edges = [
                        edge.to(device)
                        for edge in hierarchy["edge_indices"]
                    ]

                    pressure_ms_ids = [
                        indices.to(device)
                        for indices in hierarchy["pool_indices"]
                    ]

                # Synchronized batches

                for batch in loader:
                    (
                        velocity_X,
                        pressure_X,
                        velocity_Y,
                        pressure_Y,
                        dt_physical,
                        batch_simulation_ids,
                        velocity_bc,
                        velocity_dirichlet_mask,
                        pressure_bc,
                        pressure_dirichlet_mask,
                        jacobian_values,
                    ) = batch
                    if newton_weight > 0.0:
                        jacobian_values = jacobian_values.to(device)
                    velocity_X = velocity_X.to(device)
                    pressure_X = pressure_X.to(device)

                    velocity_Y = velocity_Y.to(device)
                    pressure_Y = pressure_Y.to(device)

                    dt_physical = dt_physical.to(device)

                    batch_simulation_ids = (
                        batch_simulation_ids.to(device)
                    )

                    velocity_bc = velocity_bc.to(device)

                    velocity_dirichlet_mask = (
                        velocity_dirichlet_mask.to(device)
                    )
                    pressure_bc = pressure_bc.to(device)

                    pressure_dirichlet_mask = (
                        pressure_dirichlet_mask.to(device)
                    )

                    if training:
                        velocity_optimizer.zero_grad()
                        pressure_optimizer.zero_grad()

                    # Velocity forward

                    batch_size = velocity_X.shape[0]
                    num_nodes = velocity_X.shape[1]

                    if velocity_architecture == "bsms":

                        velocity_predictions = []

                        for sample_i in range(batch_size):

                            graph = Data(
                                x=velocity_X[sample_i],
                                edge_index=edge_index,
                                edge_weight=edge_weight,
                                edge_attr=edge_attr,
                                pos=pos,
                            )

                            prediction = velocity_net(
                                graph,
                                velocity_ms_edges,
                                velocity_ms_ids,
                            )

                            velocity_predictions.append(prediction)

                        velocity_preds = torch.stack(
                            velocity_predictions,
                            dim=0,
                        )

                    else:

                        velocity_graphs = [
                            Data(
                                x=velocity_X[i],
                                edge_index=edge_index,
                                edge_weight=edge_weight,
                                edge_attr=edge_attr,
                                pos=pos,
                            )
                            for i in range(batch_size)
                        ]

                        velocity_batch = Batch.from_data_list(
                            velocity_graphs
                        )

                        velocity_output = velocity_net(
                            velocity_batch
                        )

                        velocity_preds = velocity_output.reshape(
                            batch_size,
                            num_nodes,
                            -1,
                        )

                    # Pressure forward -- NVIDIA BSMS-MGN

                    pressure_predictions = []

                    for sample_i in range(pressure_X.shape[0]):

                        graph = Data(
                            x=pressure_X[sample_i],
                            edge_index=edge_index,
                            edge_weight=edge_weight,
                            edge_attr=edge_attr,
                            pos=pos,
                        )

                        if pressure_architecture == "bsms":
                            prediction = pressure_net(
                                graph,
                                pressure_ms_edges,
                                pressure_ms_ids,
                            )

                        else:

                            prediction = pressure_net(
                                graph
                            )

                        pressure_predictions.append(
                            prediction
                        )

                    # Stack all pressure predictions only after every
                    # sample in the batch has been processed.
                    pressure_preds = torch.stack(
                        pressure_predictions,
                        dim=0,
                    )

                    # ============================================================
                    # GLOBAL HARD DIRICHLET BOUNDARY CONDITIONS
                    # Therefore the prescribed physical BC must first be converted
                    # to the corresponding nondimensional increment.
                    # ============================================================

                    state_scale_np = physics.state_scales[
                        simulation_id
                    ]

                    state_scale = torch.as_tensor(
                        state_scale_np,
                        device=device,
                        dtype=velocity_preds.dtype,
                    )

                    velocity_scale = state_scale[VELOCITY_COLUMNS]
                    pressure_scale = state_scale[PRESSURE_COLUMNS]

                    # Current state in physical units.
                    # The first three input features are the nondimensional
                    # [u_n, v_n, p_n].

                    state_now_physical = (
                        velocity_X[..., STATE_COLUMNS]
                        * state_scale
                    )

                    velocity_now_physical = (
                        state_now_physical[..., VELOCITY_COLUMNS]
                    )

                    pressure_now_physical = (
                        state_now_physical[..., PRESSURE_COLUMNS]
                    )

                    if physics.enforce_boundary_values:

                        if not predict_velocity_delta:
                            raise ValueError(
                                "Global hard velocity BC currently assumes "
                                "predict_velocity_delta=True."
                            )

                        if not predict_pressure_delta:
                            raise ValueError(
                                "Global hard pressure BC currently assumes "
                                "predict_pressure_delta=True."
                            )

                        corrected_velocity_preds = []
                        corrected_pressure_preds = []

                        for i in range(batch_size):

                            corrected_velocity_preds.append(
                                apply_velocity_dirichlet_to_delta(
                                    velocity_preds[i],
                                    velocity_now_physical[i],
                                    velocity_dirichlet_mask[i],
                                    velocity_bc[i],
                                    velocity_scale,
                                )
                            )

                            corrected_pressure_preds.append(
                                apply_pressure_dirichlet_to_delta(
                                    pressure_preds[i],
                                    pressure_now_physical[i, :, 0],
                                    pressure_dirichlet_mask[i],
                                    pressure_bc[i],
                                    pressure_scale,
                                )
                            )

                        velocity_preds = torch.stack(
                            corrected_velocity_preds,
                            dim=0,
                        )

                        pressure_preds = torch.stack(
                            corrected_pressure_preds,
                            dim=0,
                        )

                    # Targets in the same representation as the predictions.
                    # With predict_delta=True these are nondimensional increments.

                    velocity_target = velocity_Y[
                        ...,
                        VELOCITY_COLUMNS
                    ]

                    pressure_target = pressure_Y[
                        ...,
                        PRESSURE_COLUMNS
                    ]
                    # Newton/FEM loss, only when it has a weight

                    if newton_weight > 0.0:

                        if not predict_velocity_delta:
                            raise ValueError(
                                "Current FEM/Newton loss assumes "
                                "predict_velocity_delta=True."
                            )

                        if not predict_pressure_delta:
                            raise ValueError(
                                "Current FEM/Newton loss assumes "
                                "predict_pressure_delta=True."
                            )

                        # Reference: the previous state x_n, i.e. a zero
                        # increment. No gradient needed.

                        with torch.no_grad():

                            previous_metrics = fem_linearized_residual_metrics(
                                velocity_preds=torch.zeros_like(
                                    velocity_target
                                ),
                                pressure_preds=torch.zeros_like(
                                    pressure_target
                                ),
                                velocity_target=velocity_target,
                                pressure_target=pressure_target,
                                jacobian_values=jacobian_values,
                                jacobian_row=jacobian_row,
                                jacobian_col=jacobian_col,
                                jacobian_shape=jacobian_shape,
                                jacobian_dof_p=jacobian_dof_p,
                                jacobian_dof_u=jacobian_dof_u,
                                jacobian_dof_v=jacobian_dof_v,
                                state_scale=state_scale,
                                topk_fraction=newton_topk_fraction,
                            )

                        # GNN guess, outside no_grad() so the gradient
                        # reaches both networks.

                        gnn_metrics = fem_linearized_residual_metrics(
                            velocity_preds=velocity_preds,
                            pressure_preds=pressure_preds,
                            velocity_target=velocity_target,
                            pressure_target=pressure_target,
                            jacobian_values=jacobian_values,
                            jacobian_row=jacobian_row,
                            jacobian_col=jacobian_col,
                            jacobian_shape=jacobian_shape,
                            jacobian_dof_p=jacobian_dof_p,
                            jacobian_dof_u=jacobian_dof_u,
                            jacobian_dof_v=jacobian_dof_v,
                            state_scale=state_scale,
                            topk_fraction=newton_topk_fraction,
                        )

                        # Normalize relative to the previous-state guess

                        newton_mse = (
                            gnn_metrics["mse"]
                            / (
                                previous_metrics["mse"].detach()
                                + 1e-30
                            )
                        )

                        newton_topk = (
                            gnn_metrics["topk"]
                            / (
                                previous_metrics["topk"].detach()
                                + 1e-30
                            )
                        )

                        if newton_loss_type == "mse":

                            newton_loss = newton_mse

                        elif newton_loss_type == "mse_topk":

                            newton_loss = (
                                newton_mse
                                +
                                newton_topk_weight
                                * newton_topk
                            )

                        if not torch.isfinite(newton_loss):
                            raise RuntimeError(
                                "Normalized FEM/Newton loss is NaN or Inf."
                            )

                    else:

                        # Disabled: zero scalars keep the code below uniform.
                        newton_mse = torch.zeros(
                            (),
                            device=device,
                            dtype=velocity_preds.dtype,
                        )

                        newton_topk = torch.zeros_like(
                            newton_mse
                        )

                        newton_loss = torch.zeros_like(
                            newton_mse
                        )
                    
                    # Data losses

                    if data_loss_type == "mse":

                        velocity_mse = criterion(
                            velocity_preds,
                            velocity_target,
                        )

                        pressure_mse = criterion(
                            pressure_preds,
                            pressure_target,
                        )

                        velocity_topk = torch.zeros_like(
                            velocity_mse
                        )

                        pressure_topk = torch.zeros_like(
                            pressure_mse
                        )

                        velocity_data_loss = velocity_mse
                        pressure_data_loss = pressure_mse

                    elif data_loss_type == "mse_topk":

                        (
                            velocity_data_loss,
                            velocity_mse,
                            velocity_topk,
                        ) = mse_topk_loss(
                            velocity_preds,
                            velocity_target,
                            topk_fraction=topk_fraction,
                            topk_weight=topk_weight,
                            vector_error=True,
                        )

                        (
                            pressure_data_loss,
                            pressure_mse,
                            pressure_topk,
                        ) = mse_topk_loss(
                            pressure_preds,
                            pressure_target,
                            topk_fraction=topk_fraction,
                            topk_weight=topk_weight,
                            vector_error=False,
                        )

                    (
                        physics_terms,
                        normalized_physics_terms,
                    ) = physics(
                        velocity_X,
                        velocity_preds,
                        pressure_preds,
                        dt_physical,
                        batch_simulation_ids,
                        velocity_bc,
                        velocity_dirichlet_mask,
                        pressure_bc,
                        pressure_dirichlet_mask,
                    )
                    continuity_normalized = normalized_physics_terms[
                        "continuity"
                    ]

                    momentum_normalized = normalized_physics_terms[
                        "momentum"
                    ]
                    # MAX STEP: update physics weights first

                    if training and physics_weight_strategy == "dynamic":

                        weight_optimizer.zero_grad()

                        weight_objective = (
                            continuity_weight_param
                            * continuity_normalized.detach()
                            +
                            momentum_weight_param
                            * momentum_normalized.detach()
                        )

                        # Adam minimizes, so minimize -J to maximize J.
                        (-weight_objective).backward()

                        weight_optimizer.step()

                        with torch.no_grad():
                            continuity_weight_param.clamp_(min=0.0)
                            momentum_weight_param.clamp_(min=0.0)

                    # Weights used for the MIN step

                    current_continuity_weight = (
                        continuity_weight_param.detach().item()
                    )

                    current_momentum_weight = (
                        momentum_weight_param.detach().item()
                    )

                    # Build objective with UPDATED weights

                    physics_loss = (
                        continuity_weight_param.detach().clone()
                        * continuity_normalized
                        +
                        momentum_weight_param.detach().clone()
                        * momentum_normalized
                    )

                    total_loss = (
                        velocity_data_weight
                        * velocity_data_loss
                        +
                        pressure_data_weight
                        * pressure_data_loss
                        +
                        physics_loss
                        +
                        newton_weight
                        * newton_loss
                    )

                    # MIN STEP: update neural networks

                    if training:

                        total_loss.backward()

                        velocity_optimizer.step()
                        pressure_optimizer.step()

                    batch_size = velocity_X.shape[0]
                    term_sums["newton"] += (
                        newton_loss.detach().item()
                        * batch_size
                    )
                    term_sums["newton_mse"] += (
                        newton_mse.detach().item()
                        * batch_size
                    )

                    term_sums["newton_topk"] += (
                        newton_topk.detach().item()
                        * batch_size
                    )

                    term_sums["newton_weighted"] += (
                        newton_weight
                        * newton_loss.detach().item()
                        * batch_size
                    )

                    term_sums["continuity_weight"] += (
                        current_continuity_weight * batch_size
                    )

                    term_sums["momentum_weight"] += (
                        current_momentum_weight * batch_size
                    )

                    term_sums["continuity_weighted"] += (
                        current_continuity_weight
                        * continuity_normalized.detach().item()
                        * batch_size
                    )

                    term_sums["momentum_weighted"] += (
                        current_momentum_weight
                        * momentum_normalized.detach().item()
                        * batch_size
                    )
                    total_loss_sum += (
                        total_loss.item() * batch_size
                    )

                    sample_count += batch_size

                    term_sums["velocity_data"] += (
                        velocity_data_loss.item()
                        * batch_size
                    )

                    term_sums["pressure_data"] += (
                        pressure_data_loss.item()
                        * batch_size
                    )
                    term_sums["velocity_mse"] += (
                        velocity_mse.item()
                        * batch_size
                    )

                    term_sums["velocity_topk"] += (
                        velocity_topk.item()
                        * batch_size
                    )

                    term_sums["pressure_mse"] += (
                        pressure_mse.item()
                        * batch_size
                    )

                    term_sums["pressure_topk"] += (
                        pressure_topk.item()
                        * batch_size
                    )

                    for name in ("continuity", "momentum"):

                        if name in physics_terms:

                            term_sums[name] += (
                                physics_terms[name].item()
                                * batch_size
                            )
                    for name in ("continuity", "momentum"):

                        if name in normalized_physics_terms:

                            term_sums[
                                f"{name}_normalized"
                            ] += (
                                normalized_physics_terms[name].item()
                                * batch_size
                            )
        if sample_count == 0:
            raise ValueError(
                "run_epoch received an empty set of loaders."
            )
        mean_total = total_loss_sum / sample_count

        mean_terms = {
            name: value / sample_count
            for name, value in term_sums.items()
        }

        return mean_total, mean_terms

    # Epoch loop

    for epoch in range(num_epochs):

        epoch_start_time = time.perf_counter()
        # Learning rates used during this epoch.
        velocity_lr = velocity_optimizer.param_groups[0]["lr"]
        pressure_lr = pressure_optimizer.param_groups[0]["lr"]
        # TRAINING

        train_loss, train_terms = run_epoch(
            train_loaders,
            training=True,
        )

        train_loss_history.append(train_loss)

        # VALIDATION, only if a validation set exists

        if has_validation:

            val_loss, val_terms = run_epoch(
                val_loaders,
                training=False,
            )

            val_loss_history.append(val_loss)

        else:

            val_loss = None
            val_terms = None

        # REPORTING

        if verbose:

            if has_validation:

                print(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.6e} | "
                    f"Val Loss: {val_loss:.6e}"
                )

            else:

                print(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.6e} | "
                    "Val Loss: N/A (all_train)"
                )
            print(
                "           lr "
                f"velocity {velocity_lr:.3e} | "
                f"pressure {pressure_lr:.3e}"
            )
            # Training terms

            print(
                "    train: "
                f"velocity_data "
                f"{train_terms['velocity_data']:.6e} "
                f"(mse {train_terms['velocity_mse']:.6e}, "
                f"top {100 * topk_fraction:.1f}% "
                f"{train_terms['velocity_topk']:.6e}) | "
                f"pressure_data "
                f"{train_terms['pressure_data']:.6e} "
                f"(mse {train_terms['pressure_mse']:.6e}, "
                f"top {100 * topk_fraction:.1f}% "
                f"{train_terms['pressure_topk']:.6e})\n"
                f"           continuity "
                f"{train_terms['continuity']:.6e} "
                f"(norm {train_terms['continuity_normalized']:.3f}) | "
                f"momentum "
                f"{train_terms['momentum']:.6e} "
                f"(norm {train_terms['momentum_normalized']:.3f})"
            )

            print(
                "           Newton/FEM "
                f"{train_terms['newton']:.6e} "
                f"(mse {train_terms['newton_mse']:.6e}, "
                f"top {100 * newton_topk_fraction:.1f}% "
                f"{train_terms['newton_topk']:.6e}) | "
                f"weighted "
                f"{train_terms['newton_weighted']:.6e}"
            )

            print(
                "           weights "
                f"continuity "
                f"{train_terms['continuity_weight']:.6e} | "
                f"momentum "
                f"{train_terms['momentum_weight']:.6e}\n"
                "           weighted "
                f"continuity "
                f"{train_terms['continuity_weighted']:.6e} | "
                f"momentum "
                f"{train_terms['momentum_weighted']:.6e}"
            )


            # Validation terms, only when validation exists

            if has_validation:

                print(
                    "    val:   "
                    f"velocity_data "
                    f"{val_terms['velocity_data']:.6e} "
                    f"(mse {val_terms['velocity_mse']:.6e}, "
                    f"top {100 * topk_fraction:.1f}% "
                    f"{val_terms['velocity_topk']:.6e}) | "
                    f"pressure_data "
                    f"{val_terms['pressure_data']:.6e} "
                    f"(mse {val_terms['pressure_mse']:.6e}, "
                    f"top {100 * topk_fraction:.1f}% "
                    f"{val_terms['pressure_topk']:.6e})\n"
                    f"           continuity "
                    f"{val_terms['continuity']:.6e} "
                    f"(norm {val_terms['continuity_normalized']:.3f}) | "
                    f"momentum "
                    f"{val_terms['momentum']:.6e} "
                    f"(norm {val_terms['momentum_normalized']:.3f})"
                )
                print(
                    "           Newton/FEM "
                    f"{val_terms['newton']:.6e} "
                    f"(mse {val_terms['newton_mse']:.6e}, "
                    f"top {100 * newton_topk_fraction:.1f}% "
                    f"{val_terms['newton_topk']:.6e}) | "
                    f"weighted "
                    f"{val_terms['newton_weighted']:.6e}"
                )

                print(
                    "           weights "
                    f"continuity "
                    f"{val_terms['continuity_weight']:.6e} | "
                    f"momentum "
                    f"{val_terms['momentum_weight']:.6e}\n"
                    "           weighted "
                    f"continuity "
                    f"{val_terms['continuity_weighted']:.6e} | "
                    f"momentum "
                    f"{val_terms['momentum_weighted']:.6e}"
                )


        # Best model state

        if has_validation:

            # Keep the epoch with the lowest validation loss.
            if val_loss < best_val_loss:

                best_val_loss = val_loss

                best_velocity_state = copy.deepcopy(
                    velocity_net.state_dict()
                )

                best_pressure_state = copy.deepcopy(
                    pressure_net.state_dict()
                )

        else:

            # all_train: no validation, keep the last epoch.

            best_velocity_state = copy.deepcopy(
                velocity_net.state_dict()
            )

            best_pressure_state = copy.deepcopy(
                pressure_net.state_dict()
            )
        # LEARNING-RATE SCHEDULER

        if velocity_scheduler is not None:
            velocity_scheduler.step()

        if pressure_scheduler is not None:
            pressure_scheduler.step()

        epoch_time = time.perf_counter() - epoch_start_time

        if epoch_callback is not None:

            epoch_callback(
                epoch=epoch + 1,
                num_epochs=num_epochs,
                train_loss=train_loss,
                val_loss=val_loss,
                train_terms=train_terms,
                val_terms=val_terms,
                epoch_time=epoch_time,
                velocity_lr=velocity_lr,
                pressure_lr=pressure_lr,
            )

    return (
        best_velocity_state,
        best_pressure_state,
        train_loss_history,
        val_loss_history,
    )


def run_coupled_training(
    config,
    coupled_loaders,
    graph_topologies,
    bsms_hierarchies,
    physics_geometries,
    state_scales,
    physics_normalizer,
    geometry_normalizer,
    continuity_reference,
    momentum_reference,
    dt_mean,
    dt_std,
    run_dir,
    device,
    verbose,
    epoch_callback=None,
):
    """
    Control the complete coupled velocity-pressure training process.

    The training components are first created by
    setup_coupled_training(). The numerical optimization is then
    performed by train_coupled_loop(). Finally, the best model
    states are restored and saved as separate checkpoints.
    """

    # ============================================================
    # SETUP
    # ============================================================

    setup = setup_coupled_training(
        config=config,
        physics_geometries=physics_geometries,
        state_scales=state_scales,
        continuity_reference=continuity_reference,
        momentum_reference=momentum_reference,
        device=device,
    )

    velocity_model = setup["velocity_model"]
    pressure_model = setup["pressure_model"]

    velocity_optimizer = setup["velocity_optimizer"]
    pressure_optimizer = setup["pressure_optimizer"]

    velocity_scheduler = setup["velocity_scheduler"]
    pressure_scheduler = setup["pressure_scheduler"]

    criterion = setup["criterion"]
    physics = setup["physics"]

    velocity_config = setup["velocity_config"]
    pressure_config = setup["pressure_config"]

    training_config = config["training"]
    newton_config = training_config["newton_loss"]

    newton_weight = float(
        newton_config["weight"]
    )

    # ============================================================
    # TRAINING SUMMARY
    # ============================================================

    print(f"\n{'=' * 60}")
    print("COUPLED VELOCITY-PRESSURE TRAINING")

    print(
        f"Velocity: "
        f"{velocity_config['architecture']} | "
        f"{setup['velocity_num_in']} -> "
        f"{setup['velocity_num_out']}"
    )

    print(
        f"Pressure: "
        f"{pressure_config['architecture']} | "
        f"{setup['pressure_num_in']} -> "
        f"{setup['pressure_num_out']}"
    )

    print(
        "Loss: "
        f"{training_config['velocity_data_weight']:g} L_u + "
        f"{training_config['pressure_data_weight']:g} L_p + "
        f"{training_config['continuity_weight']:g} L_cont + "
        f"{training_config['momentum_weight']:g} L_mom + "
        f"{newton_weight:g} L_Newton"
    )

    print(
        "Physics weight strategy: "
        f"{training_config['physics_weight_strategy']}"
    )

    print(
        "Data loss: "
        f"{training_config['data_loss']['type']} | "
        f"top-k fraction="
        f"{training_config['data_loss']['topk_fraction']:.3f} | "
        f"top-k weight="
        f"{training_config['data_loss']['topk_weight']:.3f}"
    )

    if newton_weight > 0.0:
        print(
            "Newton loss: "
            f"{newton_config['type']} | "
            f"top-k fraction="
            f"{newton_config['topk_fraction']:.3f} | "
            f"top-k weight="
            f"{newton_config['topk_weight']:.3f}"
        )

    if training_config["physics_weight_strategy"] == "dynamic":
        print(
            "Dynamic weight learning rate: "
            f"{training_config['dynamic_weight_learning_rate']:.3e}"
        )

    print("=" * 60)

    # ============================================================
    # TRAINING LOOP
    # ============================================================

    (
        velocity_state,
        pressure_state,
        train_history,
        val_history,
    ) = train_coupled_loop(
        velocity_model,
        pressure_model,
        coupled_loaders["train"],
        coupled_loaders["val"],
        graph_topologies,
        bsms_hierarchies,
        physics,
        velocity_optimizer,
        pressure_optimizer,
        criterion,
        training_config["num_epochs"],
        device,

        # Architectures
        velocity_architecture=velocity_config["architecture"],
        pressure_architecture=pressure_config["architecture"],

        # Schedulers
        velocity_scheduler=velocity_scheduler,
        pressure_scheduler=pressure_scheduler,

        # Data loss
        velocity_data_weight=training_config["velocity_data_weight"],
        pressure_data_weight=training_config["pressure_data_weight"],
        data_loss_type=training_config["data_loss"]["type"],
        topk_fraction=training_config["data_loss"]["topk_fraction"],
        topk_weight=training_config["data_loss"]["topk_weight"],

        # Physics loss
        physics_weight_strategy=training_config["physics_weight_strategy"],
        continuity_weight=training_config["continuity_weight"],
        momentum_weight=training_config["momentum_weight"],
        dynamic_weight_learning_rate=training_config["dynamic_weight_learning_rate"],

        # Newton loss
        newton_weight=newton_weight,
        newton_loss_type=newton_config["type"],
        newton_topk_fraction=newton_config["topk_fraction"],
        newton_topk_weight=newton_config["topk_weight"],

        # Prediction representation
        predict_velocity_delta=velocity_config["predict_delta"],
        predict_pressure_delta=pressure_config["predict_delta"],

        verbose=verbose,
        epoch_callback=epoch_callback,
    )

    # ============================================================
    # RESTORE BEST JOINT STATE
    # ============================================================

    velocity_model.load_state_dict(
        velocity_state
    )

    pressure_model.load_state_dict(
        pressure_state
    )

    velocity_model.eval()
    pressure_model.eval()

    # ============================================================
    # SAVE TRAINING OUTPUTS
    # ============================================================

    trained = save_coupled_training_outputs(
        config=config,
        velocity_model=velocity_model,
        pressure_model=pressure_model,
        velocity_config=velocity_config,
        pressure_config=pressure_config,
        velocity_num_in=setup["velocity_num_in"],
        velocity_num_out=setup["velocity_num_out"],
        pressure_num_in=setup["pressure_num_in"],
        pressure_num_out=setup["pressure_num_out"],
        physics_normalizer=physics_normalizer,
        geometry_normalizer=geometry_normalizer,
        dt_mean=dt_mean,
        dt_std=dt_std,
        train_history=train_history,
        val_history=val_history,
        run_dir=run_dir,
    )

    return trained