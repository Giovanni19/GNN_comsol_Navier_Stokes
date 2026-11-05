from torch.utils.data import DataLoader

from .features import build_features, build_node_type_features
from .graphs import create_coupled_dataset

# Loaders: one tensor loader per simulation (one mesh hierarchy each).


def build_coupled_loaders(
    normalized,
    config,
):
    """
    Synchronized velocity-pressure loaders, one per simulation (a BSMS
    batch must share one mesh hierarchy). Each item is one transition
    for both networks.
    """
    velocity_network = config["networks"]["velocity"]
    pressure_network = config["networks"]["pressure"]

    batch_size = config["training"]["batch_size"]

    coupled_loaders = {
        "train": {},
        "val": {},
        "test": {},
    }

    # Targets used by the two networks

    velocity_target_key = (
        "delta"
        if velocity_network["predict_delta"]
        else "Y"
    )

    pressure_target_key = (
        "delta"
        if pressure_network["predict_delta"]
        else "Y"
    )

    # Build one synchronized dataset per simulation

    for split_name in ("train", "val", "test"):

        for simulation in normalized[split_name]:

            simulation_id = simulation["simulation_id"]
            node_type_features = None

            if (
                velocity_network["use_node_type_features"]
                or pressure_network["use_node_type_features"]
            ):
                wall_mask = simulation.get("wall_mask")
                inlet_mask = simulation.get("inlet_mask")
                outlet_mask = simulation.get("outlet_mask")

                if (
                    wall_mask is None
                    or inlet_mask is None
                    or outlet_mask is None
                ):
                    raise ValueError(
                        f"Simulation {simulation_id} requires node-type features, "
                        "but wall_mask, inlet_mask or outlet_mask is missing."
                    )

                node_type_features = build_node_type_features(
                    wall_mask,
                    inlet_mask,
                    outlet_mask,
                )
            # Velocity input

            velocity_features = build_features(
                velocity_network["features"],
                simulation["X"],
                simulation["dt"],
                physics_features=(
                    simulation["physics_features"]
                    if velocity_network["use_physics_features"]
                    else None
                ),
                geometry_features=(
                    simulation["geometry_features"]
                    if velocity_network["use_geometry_features"]
                    else None
                ),
                node_type_features=(
                    node_type_features
                    if velocity_network["use_node_type_features"]
                    else None
                ),
            )
            # Pressure input

            pressure_features = build_features(
                pressure_network["features"],
                simulation["X"],
                simulation["dt"],
                physics_features=(
                    simulation["physics_features"]
                    if pressure_network["use_physics_features"]
                    else None
                ),
                geometry_features=(
                    simulation["geometry_features"]
                    if pressure_network["use_geometry_features"]
                    else None
                ),
                node_type_features=(
                    node_type_features
                    if pressure_network["use_node_type_features"]
                    else None
                ),
            )

            # Targets

            velocity_target = simulation[
                velocity_target_key
            ]

            pressure_target = simulation[
                pressure_target_key
            ]

            # Synchronized dataset

            dataset = create_coupled_dataset(
                velocity_features,
                pressure_features,
                velocity_target,
                pressure_target,
                simulation_id,
                simulation["dt_physical"],

                velocity_bc_target=simulation.get(
                    "velocity_bc_target"
                ),
                velocity_dirichlet_mask=simulation.get(
                    "velocity_dirichlet_mask"
                ),

                pressure_bc_target=simulation.get(
                    "pressure_bc_target"
                ),
                pressure_dirichlet_mask=simulation.get(
                    "pressure_dirichlet_mask"
                ),

                # (num_samples, nnz); the COO pattern is in
                # graph_topologies.
                jacobian_values=simulation.get(
                    "jacobian_values"
                ),
            )

            coupled_loaders[
                split_name
            ][simulation_id] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split_name == "train"),
            )

            print(
                f"Coupled {split_name} | "
                f"simulation {simulation_id}: "
                f"{len(dataset)} samples | "
                f"{simulation['X'].shape[1]} nodes | "
                f"velocity {velocity_features.shape[-1]} features | "
                f"pressure {pressure_features.shape[-1]} features"
            )

    return coupled_loaders