"""
Experiment-level data preparation.

This module selects the configured datasets, loads the simulations,
and creates the train/validation/test split.
"""

from types import SimpleNamespace

from gnn_comsol import data as gdata


def dataset_paths(config):
    """Return the dataset paths configured for the experiment."""

    dataset = config["dataset"]

    if dataset.get("paths"):
        return list(dataset["paths"])

    return [dataset["path"]]


def load_simulations(config):
    """Load all simulations configured for the experiment."""

    simulations = gdata.load_simulations(
        dataset_paths(config),
        skip_initial=config["dataset"]["skip_initial"],
    )

    print(
        f"\nLoaded {len(simulations)} simulations "
        f"({config['dataset']['skip_initial']} initial snapshot(s) "
        f"dropped per simulation)"
    )

    for simulation in simulations:

        dt = simulation.delta_t

        print(
            f"  Simulation {simulation.simulation_id}: "
            f"{simulation.num_samples} samples | "
            f"{simulation.num_nodes} nodes | "
            f"{simulation.num_edges} edges | "
            f"dt mean={dt.mean():.4e} | "
            f"min={dt.min():.4e} | "
            f"max={dt.max():.4e} | "
            f"{simulation.file_path}"
        )

    return simulations


def split_simulations(simulations, config):
    """Create the configured train/validation/test split."""

    split = config["split"]
    mode = split["mode"]

    if mode == "all_train":

        splits = SimpleNamespace(
            train=list(simulations),
            val=[],
            test=[],
        )

    elif mode == "simulation":

        splits = gdata.split_simulations(
            simulations,
            train_ids=split["train_simulations"],
            val_ids=split["val_simulations"],
            test_ids=split["test_simulations"],
        )

    else:

        splits = gdata.split_simulations_by_sample(
            simulations,
            mode=mode,
            train_fraction=split["train_fraction"],
            val_fraction=split["val_fraction"],
            gap=split["gap"],
            seed=config["seed"],
        )

    print(f"\nSplit ({mode}):")

    for name in ("train", "val", "test"):

        block = getattr(splits, name)

        samples = sum(
            simulation.num_samples
            for simulation in block
        )

        print(
            f"\n{name.upper()} "
            f"({len(block)} simulation(s), {samples} samples)"
        )

        for simulation in block:

            print(
                f"  Simulation {simulation.simulation_id}: "
                f"{simulation.num_nodes} nodes | "
                f"{simulation.num_samples} samples | "
                f"{simulation.file_path}"
            )

    return splits