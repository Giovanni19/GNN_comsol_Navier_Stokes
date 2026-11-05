"""
Output utilities for coupled velocity-pressure training.

This module saves trained model checkpoints and training-history plots.
"""

from gnn_comsol import data as gdata
from gnn_comsol import plots
from gnn_comsol.checkpoints import save_checkpoint


def save_coupled_training_outputs(
    config,
    velocity_model,
    pressure_model,
    velocity_config,
    pressure_config,
    velocity_num_in,
    velocity_num_out,
    pressure_num_in,
    pressure_num_out,
    physics_normalizer,
    geometry_normalizer,
    dt_mean,
    dt_std,
    train_history,
    val_history,
    run_dir,
):
    """
    Save the trained velocity and pressure checkpoints and the
    coupled training-loss plot.

    Returns
    -------
    dict
        Trained model information used by the rest of the pipeline.
    """

    training_config = config["training"]

    network_results = {
        "velocity": {
            "model": velocity_model,
            "config": velocity_config,
            "num_in": velocity_num_in,
            "num_out": velocity_num_out,
            "columns": gdata.VELOCITY_COLUMNS,
        },
        "pressure": {
            "model": pressure_model,
            "config": pressure_config,
            "num_in": pressure_num_in,
            "num_out": pressure_num_out,
            "columns": gdata.PRESSURE_COLUMNS,
        },
    }

    trained = {}

    for network_name in ("velocity", "pressure"):

        network = config["networks"][network_name]
        result = network_results[network_name]

        uses_physics = network["use_physics_features"]
        uses_geometry = network["use_geometry_features"]
        predicts_delta = network["predict_delta"]

        checkpoint_path = run_dir / f"{network_name}.pth"

        save_checkpoint(
            checkpoint_path,
            result["model"],
            physics_normalizer=(
                physics_normalizer if uses_physics else None
            ),
            geometry_normalizer=(
                geometry_normalizer if uses_geometry else None
            ),
            dt_normalization=(dt_mean, dt_std),
            metadata={
                "experiment": config["name"],
                "network": network_name,
                "predicts": network["predicts"],
                "features": network["features"],
                "use_physics_features": uses_physics,
                "physics_feature_names": gdata.PHYSICS_FEATURE_NAMES if uses_physics else [],
                "predict_delta": predicts_delta,
                "delta_scaling": "physical_nondimensionalization",
                "use_geometry_features": uses_geometry,
                "use_node_type_features": network["use_node_type_features"],
                "node_type_feature_names": list(gdata.NODE_TYPE_FEATURE_NAMES) if network["use_node_type_features"] else [],
                "geometry_feature_names": gdata.GEOMETRY_FEATURE_NAMES if uses_geometry else [],
                "coupled_training": True,
                "num_in": result["num_in"],
                "num_out": result["num_out"],
                "split_mode": config["split"]["mode"],
                "num_epochs": training_config["num_epochs"],
                "batch_size": training_config["batch_size"],
                "architecture": result["config"]["architecture"],
                "num_neurons": result["config"]["num_neurons"],
                "num_layers": result["config"]["num_layers"],
                "dropout": result["config"]["dropout"],
                "learning_rate": result["config"]["learning_rate"],
                "weight_decay": result["config"]["weight_decay"],
                "processor_size": result["config"].get("processor_size"),
                "num_layers_node_processor": result["config"].get("num_layers_node_processor"),
                "num_layers_edge_processor": result["config"].get("num_layers_edge_processor"),
                "num_layers_node_encoder": result["config"].get("num_layers_node_encoder"),
                "num_layers_edge_encoder": result["config"].get("num_layers_edge_encoder"),
                "num_layers_node_decoder": result["config"].get("num_layers_node_decoder"),
                "aggregation": result["config"].get("aggregation"),
                "num_mesh_levels": result["config"].get("num_mesh_levels"),
                "num_layers_bistride": result["config"].get("num_layers_bistride"),
                "bistride_unet_levels": result["config"].get("bistride_unet_levels"),
            },
        )

        trained[network_name] = {
            "model": result["model"],
            "columns": result["columns"],
            "features": network["features"],
            "use_physics_features": uses_physics,
            "use_node_type_features": network["use_node_type_features"],
            "use_geometry_features": uses_geometry,
            "predict_delta": predicts_delta,
        }

        print(
            f"Saved coupled {network_name} checkpoint: "
            f"{checkpoint_path.name}"
        )

    plots.plot_loss(
        train_history,
        val_history,
        path=run_dir / "coupled_loss.png",
        title="coupled velocity-pressure",
    )

    trained["_training_summary"] = {
        "train_history": train_history,
        "val_history": val_history,
    }

    return trained