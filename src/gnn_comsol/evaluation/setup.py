"""
Setup utilities for evaluating a trained run.
"""

import json

import numpy as np

from gnn_comsol import data as gdata
from gnn_comsol.checkpoints import read_checkpoint
from gnn_comsol.models import build_model
from gnn_comsol.graph.hierarchy import build_bsms_hierarchy

def read_run(run_dir):
    """The resolved config and the metrics of a finished run."""

    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"

    for path in (config_path, metrics_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing: {run_dir} does not look like a "
                "finished run directory."
            )

    with open(config_path) as handle:
        config = json.load(handle)

    with open(metrics_path) as handle:
        metrics = json.load(handle)

    return config, metrics


def choose_test_simulation(config, metrics, simulation_id):
    """
    Which .mat to evaluate and its simulation id, taken from the run's
    test block so a training geometry is never reported as a test one.
    """

    dataset = config["dataset"]

    paths = dataset.get("paths") or [dataset["path"]]

    test_ids = metrics["test_simulations"]

    if not test_ids:
        raise ValueError(
            "This run has no test simulations recorded in metrics.json."
        )

    if simulation_id is None:
        simulation_id = test_ids[0]

    elif simulation_id not in test_ids:
        raise ValueError(
            f"Simulation {simulation_id} was not in the test block of "
            f"this run: the held-out simulations are {test_ids}. "
            "Evaluating on a simulation the model was trained on does "
            "not measure generalisation."
        )

    if simulation_id >= len(paths):
        raise ValueError(
            f"Simulation {simulation_id} is out of range: the run used "
            f"{len(paths)} dataset(s)."
        )

    return paths[simulation_id], simulation_id



def held_out_samples(raw, config, split_mode, skip_initial=None):
    """
    The samples of `raw` that the run's split kept out of training.

    skip_initial is the one `raw` was loaded with (default: the
    training one). The split is recomputed on the samples the run
    actually had, then shifted to the indices of `raw`; transitions the
    run skipped are not held out.
    """

    if split_mode == "simulation":
        return raw, np.arange(raw.num_samples)

    if split_mode == "all_train":
        raise ValueError(
            "An all_train run has no internal held-out samples. "
            "Evaluate it with --dataset on an external unseen dataset."
        )

    split = config["split"]

    training_skip = config["dataset"].get("skip_initial", 0)

    if skip_initial is None:
        skip_initial = training_skip

    num_transitions = raw.num_samples + skip_initial

    _, _, test_indices = gdata.compute_split_indices(
        num_transitions - training_skip,
        mode=split_mode,
        train_fraction=split["train_fraction"],
        val_fraction=split["val_fraction"],
        gap=split["gap"],
        seed=config["seed"]
    )

    # Training sample index -> index in `raw`
    test_indices = test_indices + training_skip - skip_initial
    test_indices = test_indices[test_indices >= 0]

    return gdata.subset_simulation(raw, test_indices), test_indices


def build_from_checkpoint(checkpoint_path, device):
    """
    Rebuild a model from its own checkpoint.

    The metadata carries everything the constructor needs, so nothing
    here has to know which experiment produced the file.
    """

    # First pass: metadata and scalings, before the model exists
    bundle = read_checkpoint(checkpoint_path, device=device)

    metadata = bundle.metadata

    architecture = metadata["architecture"]

    model_kwargs = {
        "architecture": architecture,
        "num_in": metadata["num_in"],
        "num_out": metadata["num_out"],
        "num_neurons": metadata["num_neurons"],
        "num_layers": metadata["num_layers"],
        "dropout": metadata["dropout"],
    }

    if architecture in {"meshgraphnet", "bsms"}:

        model_kwargs.update(
            {
                "processor_size": metadata["processor_size"],
                "edge_dim": 3,

                "num_layers_node_processor": metadata[
                    "num_layers_node_processor"
                ],
                "num_layers_edge_processor": metadata[
                    "num_layers_edge_processor"
                ],

                "num_layers_node_encoder": metadata[
                    "num_layers_node_encoder"
                ],
                "num_layers_edge_encoder": metadata[
                    "num_layers_edge_encoder"
                ],
                "num_layers_node_decoder": metadata[
                    "num_layers_node_decoder"
                ],

                "aggregation": metadata["aggregation"],
            }
        )

    if architecture == "bsms":

        model_kwargs.update(
            {
                "num_mesh_levels": metadata["num_mesh_levels"],
                "num_layers_bistride": metadata[
                    "num_layers_bistride"
                ],
                "bistride_unet_levels": metadata[
                    "bistride_unet_levels"
                ],
                "pos_dim": 2,
            }
        )

    model = build_model(**model_kwargs).to(device)

    # Second pass: the weights
    bundle = read_checkpoint(
        checkpoint_path,
        model=model,
        device=device
    )

    bundle.model.eval()

    return bundle

def agree(first_values, second_values, what, first, second):

    if not np.allclose(first_values, second_values):
        raise ValueError(
            f"{first} and {second} were saved with different {what}, "
            "so they cannot have come from the same run."
        )


def prepare_evaluation_data(
    config,
    metrics,
    dataset_override=None,
    simulation_id=None,
    skip_initial=0,
    held_out_only=False,
):
    """
    Load and prepare the dataset used for evaluation.

    Returns
    -------
    evaluation
        Simulation data actually passed to the evaluator.

    dataset_path
        Path of the evaluated .mat file.

    simulation_id
        Simulation identifier.

    time
        Physical target time of every evaluated transition.

    indices
        Sample indices relative to the loaded trajectory.

    report_mask
        Boolean held-out mask, or None when every evaluated sample
        is held out.

    inference_skip_initial
        Number of transitions skipped before evaluation.
    """

    inference_skip_initial = skip_initial

    # ------------------------------------------------------------
    # Select dataset
    # ------------------------------------------------------------

    if dataset_override is not None:

        dataset_path = dataset_override

        if simulation_id is None:
            simulation_id = 0

        print(
            f"\nEvaluating {dataset_path} "
            "(external dataset supplied on the command line)"
        )

        external_dataset = True

    else:

        dataset_path, simulation_id = choose_test_simulation(
            config,
            metrics,
            simulation_id,
        )

        print(
            f"\nEvaluating held-out simulation {simulation_id}"
        )

        external_dataset = False

    # ------------------------------------------------------------
    # Load trajectory
    # ------------------------------------------------------------

    training_skip_initial = config["dataset"].get(
        "skip_initial",
        0,
    )

    print(
        f"\nTraining skip_initial  : {training_skip_initial}\n"
        f"Inference skip_initial : {inference_skip_initial}"
    )

    raw = gdata.load_data(
        dataset_path,
        skip_initial=inference_skip_initial,
        simulation_id=simulation_id,
        load_jacobian=False,
    )

    # Full trajectory is needed only to reconstruct absolute target times.
    raw_full = gdata.load_data(
        dataset_path,
        skip_initial=0,
        simulation_id=simulation_id,
        load_jacobian=False,
    )

    full_target_time = np.cumsum(raw_full.delta_t)

    time_axis = full_target_time[
        inference_skip_initial:
        inference_skip_initial + raw.num_samples
    ]

    if len(time_axis) != raw.num_samples:
        raise ValueError(
            "Could not reconstruct the physical target times after "
            f"skip_initial={inference_skip_initial}: "
            f"{raw.num_samples} samples but {len(time_axis)} times."
        )

    print(
        f"Physical target times after "
        f"skip_initial={inference_skip_initial}: "
        f"{time_axis[0]:.6e} -> {time_axis[-1]:.6e} s"
    )

    # ------------------------------------------------------------
    # Determine held-out samples
    # ------------------------------------------------------------

    if external_dataset:

        held_out_indices = np.arange(
            raw.num_samples,
            dtype=int,
        )

    else:

        _, held_out_indices = held_out_samples(
            raw,
            config,
            metrics["split_mode"],
            skip_initial=inference_skip_initial,
        )

    held_out_mask = np.zeros(
        raw.num_samples,
        dtype=bool,
    )

    held_out_mask[held_out_indices] = True

    # ------------------------------------------------------------
    # Select samples actually evaluated
    # ------------------------------------------------------------

    if held_out_only:

        evaluation = gdata.subset_simulation(
            raw,
            held_out_indices,
        )

        indices = held_out_indices
        time = time_axis[held_out_indices]

        # Every evaluated sample is held out.
        report_mask = None

    else:

        evaluation = raw
        indices = np.arange(raw.num_samples)
        time = time_axis
        report_mask = held_out_mask

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    print(
        f"  {dataset_path}\n"
        f"  {raw.num_nodes} nodes | "
        f"{raw.num_edges} edges | "
        f"{raw.num_samples} samples, of which "
        f"{int(held_out_mask.sum())} were held out by the "
        f"{metrics['split_mode']!r} split"
    )

    if not held_out_only and not held_out_mask.all():
        print(
            f"  Evaluating all {raw.num_samples} transitions: "
            "some samples were used during training. "
            "Generalisation metrics are reported separately."
        )

    return (
        evaluation,
        dataset_path,
        simulation_id,
        time,
        indices,
        report_mask,
        inference_skip_initial,
    )

def build_evaluation_hierarchies(
    evaluation,
    velocity,
    pressure,
    device,
):
    """
    Build the BSMS hierarchy required by each evaluated network.

    Velocity and pressure are handled independently, so either,
    both, or neither network may use BSMS. If both use BSMS,
    they may also use different numbers of mesh levels.

    Returns
    -------
    dict
        Dictionary with keys "velocity" and "pressure".
        The value is None when that network does not use BSMS.
    """

    hierarchies = {}

    for network_name, bundle in (
        ("velocity", velocity),
        ("pressure", pressure),
    ):

        architecture = bundle.metadata["architecture"]

        if architecture != "bsms":
            hierarchies[network_name] = None
            continue

        num_mesh_levels = bundle.metadata["num_mesh_levels"]

        hierarchy = build_bsms_hierarchy(
            evaluation.edge_index,
            evaluation.num_nodes,
            evaluation.pos,
            num_mesh_levels,
        )

        hierarchies[network_name] = {
            "edge_indices": [
                edge.to(device)
                for edge in hierarchy["edge_indices"]
            ],
            "pool_indices": [
                indices.to(device)
                for indices in hierarchy["pool_indices"]
            ],
        }

    return hierarchies


def load_evaluation_models(
    run_dir,
    device,
):
    """
    Load the trained velocity and pressure networks for evaluation.

    The models are rebuilt from their checkpoint metadata and their
    saved time-step normalizations are checked for consistency.

    Parameters
    ----------
    run_dir : Path
        Directory containing velocity.pth and pressure.pth.

    device : torch.device
        Device on which the models are loaded.

    dt_normalization : tuple[float, float] or None
        Optional command-line override for old checkpoints that do not
        contain the time-step normalization.

    Returns
    -------
    velocity
        Loaded velocity checkpoint bundle.

    pressure
        Loaded pressure checkpoint bundle.

    dt_mean : float
        Time-step normalization mean used during evaluation.
    """

    velocity_path = run_dir / "velocity.pth"
    pressure_path = run_dir / "pressure.pth"

    velocity = build_from_checkpoint(
        velocity_path,
        device,
    )

    pressure = build_from_checkpoint(
        pressure_path,
        device,
    )

    print("\nVelocity metadata:", velocity.metadata)
    print("\nPressure metadata:", pressure.metadata)

    return velocity, pressure