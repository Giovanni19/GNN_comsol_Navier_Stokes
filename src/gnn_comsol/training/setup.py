"""
Setup utilities for coupled velocity-pressure training.

This module builds and initializes the velocity and pressure models,
optimizers, learning-rate schedulers, data-loss criterion, and coupled
physics loss required by the training loop.
"""

import torch.nn as nn
import torch.optim as optim

from gnn_comsol import data as gdata
from gnn_comsol.models import (
    build_model,
    initialize_model,
    report_initialization,
)

from .losses import CoupledPhysicsLoss


def setup_coupled_training(
    config,
    physics_geometries,
    state_scales,
    continuity_reference,
    momentum_reference,
    device,
):
    """
    Build everything required by the coupled training loop.

    This function only prepares the training components. It does not
    execute the training loop.

    Returns
    -------
    dict
        Models, optimizers, schedulers, criterion, physics loss,
        network configurations, and input/output dimensions.
    """

    velocity_config = config["networks"]["velocity"]
    pressure_config = config["networks"]["pressure"]
    training_config = config["training"]

    # ============================================================
    # INPUT / OUTPUT SIZES
    # ============================================================

    velocity_num_in = gdata.features_size(
        velocity_config["features"],
        use_physics_features=velocity_config[
            "use_physics_features"
        ],
        use_geometry_features=velocity_config[
            "use_geometry_features"
        ],
        use_node_type_features=velocity_config[
            "use_node_type_features"
        ],
    )

    pressure_num_in = gdata.features_size(
        pressure_config["features"],
        use_physics_features=pressure_config[
            "use_physics_features"
        ],
        use_geometry_features=pressure_config[
            "use_geometry_features"
        ],
        use_node_type_features=pressure_config[
            "use_node_type_features"
        ],
    )

    velocity_num_out = 2
    pressure_num_out = 1

    # ============================================================
    # VELOCITY MODEL
    # ============================================================

    velocity_kwargs = {
        "architecture": velocity_config["architecture"],
        "num_in": velocity_num_in,
        "num_out": velocity_num_out,
        "num_neurons": velocity_config["num_neurons"],
        "num_layers": velocity_config["num_layers"],
        "dropout": velocity_config["dropout"],
    }

    if velocity_config["architecture"] == "meshgraphnet":
        velocity_kwargs.update(
            {
                "processor_size": velocity_config["processor_size"],
                "edge_dim": 3,
                "num_layers_node_processor": velocity_config[
                    "num_layers_node_processor"
                ],
                "num_layers_edge_processor": velocity_config[
                    "num_layers_edge_processor"
                ],
                "num_layers_node_encoder": velocity_config[
                    "num_layers_node_encoder"
                ],
                "num_layers_edge_encoder": velocity_config[
                    "num_layers_edge_encoder"
                ],
                "num_layers_node_decoder": velocity_config[
                    "num_layers_node_decoder"
                ],
                "aggregation": velocity_config["aggregation"],
            }
        )

    elif velocity_config["architecture"] == "bsms":
        velocity_kwargs.update(
            {
                "processor_size": velocity_config["processor_size"],
                "edge_dim": 3,
                "num_mesh_levels": velocity_config[
                    "num_mesh_levels"
                ],
                "num_layers_bistride": velocity_config[
                    "num_layers_bistride"
                ],
                "bistride_unet_levels": velocity_config[
                    "bistride_unet_levels"
                ],
                "num_layers_node_processor": velocity_config[
                    "num_layers_node_processor"
                ],
                "num_layers_edge_processor": velocity_config[
                    "num_layers_edge_processor"
                ],
                "num_layers_node_encoder": velocity_config[
                    "num_layers_node_encoder"
                ],
                "num_layers_edge_encoder": velocity_config[
                    "num_layers_edge_encoder"
                ],
                "num_layers_node_decoder": velocity_config[
                    "num_layers_node_decoder"
                ],
                "aggregation": velocity_config["aggregation"],
                "pos_dim": 2,
            }
        )

    velocity_model = build_model(
        **velocity_kwargs
    ).to(device)

    initialize_model(
        velocity_model,
        velocity_config.get("initialization", "default"),
        num_out=velocity_num_out,
    )

    report_initialization(
        velocity_model,
        "Velocity",
    )

    # ============================================================
    # PRESSURE MODEL
    # ============================================================

    pressure_kwargs = {
        "architecture": pressure_config["architecture"],
        "num_in": pressure_num_in,
        "num_out": pressure_num_out,
        "num_neurons": pressure_config["num_neurons"],
        "num_layers": pressure_config["num_layers"],
        "dropout": pressure_config["dropout"],
    }

    if pressure_config["architecture"] == "meshgraphnet":
        pressure_kwargs.update(
            {
                "processor_size": pressure_config["processor_size"],
                "edge_dim": 3,
                "num_layers_node_processor": pressure_config[
                    "num_layers_node_processor"
                ],
                "num_layers_edge_processor": pressure_config[
                    "num_layers_edge_processor"
                ],
                "num_layers_node_encoder": pressure_config[
                    "num_layers_node_encoder"
                ],
                "num_layers_edge_encoder": pressure_config[
                    "num_layers_edge_encoder"
                ],
                "num_layers_node_decoder": pressure_config[
                    "num_layers_node_decoder"
                ],
                "aggregation": pressure_config["aggregation"],
            }
        )

    elif pressure_config["architecture"] == "bsms":
        pressure_kwargs.update(
            {
                "processor_size": pressure_config["processor_size"],
                "edge_dim": 3,
                "num_mesh_levels": pressure_config[
                    "num_mesh_levels"
                ],
                "num_layers_bistride": pressure_config[
                    "num_layers_bistride"
                ],
                "bistride_unet_levels": pressure_config[
                    "bistride_unet_levels"
                ],
                "num_layers_node_processor": pressure_config[
                    "num_layers_node_processor"
                ],
                "num_layers_edge_processor": pressure_config[
                    "num_layers_edge_processor"
                ],
                "num_layers_node_encoder": pressure_config[
                    "num_layers_node_encoder"
                ],
                "num_layers_edge_encoder": pressure_config[
                    "num_layers_edge_encoder"
                ],
                "num_layers_node_decoder": pressure_config[
                    "num_layers_node_decoder"
                ],
                "aggregation": pressure_config["aggregation"],
                "pos_dim": 2,
            }
        )

    pressure_model = build_model(
        **pressure_kwargs
    ).to(device)

    initialize_model(
        pressure_model,
        pressure_config.get("initialization", "default"),
        num_out=pressure_num_out,
    )

    report_initialization(
        pressure_model,
        "Pressure",
    )

    # ============================================================
    # OPTIMIZERS
    # ============================================================

    velocity_optimizer = optim.Adam(
        velocity_model.parameters(),
        lr=velocity_config["learning_rate"],
        weight_decay=velocity_config["weight_decay"],
    )

    pressure_optimizer = optim.Adam(
        pressure_model.parameters(),
        lr=pressure_config["learning_rate"],
        weight_decay=pressure_config["weight_decay"],
    )

    # ============================================================
    # LEARNING-RATE SCHEDULERS
    # ============================================================

    scheduler_config = training_config["lr_scheduler"]

    velocity_scheduler = None
    pressure_scheduler = None

    if scheduler_config["type"] == "step":

        velocity_scheduler = optim.lr_scheduler.StepLR(
            velocity_optimizer,
            step_size=scheduler_config["step_size"],
            gamma=scheduler_config["gamma"],
        )

        pressure_scheduler = optim.lr_scheduler.StepLR(
            pressure_optimizer,
            step_size=scheduler_config["step_size"],
            gamma=scheduler_config["gamma"],
        )

    elif scheduler_config["type"] == "exponential":

        velocity_scheduler = optim.lr_scheduler.ExponentialLR(
            velocity_optimizer,
            gamma=scheduler_config["gamma"],
        )

        pressure_scheduler = optim.lr_scheduler.ExponentialLR(
            pressure_optimizer,
            gamma=scheduler_config["gamma"],
        )

    # ============================================================
    # DATA-LOSS CRITERION
    # ============================================================

    criterion = nn.MSELoss()

    # ============================================================
    # COUPLED PHYSICS LOSS
    # ============================================================

    physics = CoupledPhysicsLoss(
        geometries=physics_geometries,
        state_scales=state_scales,
        predict_velocity_delta=velocity_config[
            "predict_delta"
        ],
        predict_pressure_delta=pressure_config[
            "predict_delta"
        ],
        continuity_reference=continuity_reference,
        momentum_reference=momentum_reference,
        enforce_boundary_values=training_config[
            "enforce_boundary_values"
        ],
    )

    # ============================================================
    # RETURN TRAINING COMPONENTS
    # ============================================================

    return {
        "velocity_model": velocity_model,
        "pressure_model": pressure_model,

        "velocity_optimizer": velocity_optimizer,
        "pressure_optimizer": pressure_optimizer,

        "velocity_scheduler": velocity_scheduler,
        "pressure_scheduler": pressure_scheduler,

        "criterion": criterion,
        "physics": physics,

        "velocity_config": velocity_config,
        "pressure_config": pressure_config,

        "velocity_num_in": velocity_num_in,
        "velocity_num_out": velocity_num_out,

        "pressure_num_in": pressure_num_in,
        "pressure_num_out": pressure_num_out,
    }