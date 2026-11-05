"""
Experiment configuration.

An experiment is a YAML file: networks, encodings and hyperparameters
are data, not code.

Schema
------
name: str
seed: int
dataset:
  path: str              # one simulation
  paths: [str, ...]      # or several; exactly one of the two
  skip_initial: int      # snapshots dropped from the start
split:
  mode: temporal | random | simulation | all_train
  train_fraction: float
  val_fraction: float    # temporal / random only
  gap: int               # temporal only
training:
  num_epochs: int
  batch_size: int
  coupled: true          # the only supported training mode
  velocity_data_weight: float
  pressure_data_weight: float
  data_loss: {type: mse | mse_topk, topk_fraction, topk_weight}
  lr_scheduler: {type: none | step | exponential, step_size, gamma}
  continuity_weight: float
  momentum_weight: float
  physics_weight_strategy: fixed | dynamic
  dynamic_weight_learning_rate: float
  enforce_boundary_values: bool
  newton_loss: {weight, type: mse | mse_topk, topk_fraction, topk_weight}
networks:                    # exactly `velocity` and `pressure`
  <velocity | pressure>:
    predicts: velocity | pressure      # same as the network name
    features: time | time_fourier
    architecture: gcn | gcn_virtual_node | meshgraphnet   # velocity
                | bsms                                     # pressure
    initialization: default | zero | zero_output
    num_neurons: int 
    num_layers: int 
    dropout: float 
    learning_rate: float 
    weight_decay: float 
    # meshgraphnet and bsms:
    processor_size: int
    num_layers_node_processor, num_layers_edge_processor: int
    num_layers_node_encoder, num_layers_edge_encoder: int
    num_layers_node_decoder: int
    aggregation: sum | mean
    # bsms only:
    num_mesh_levels: int
    num_layers_bistride: int
    bistride_unet_levels: int
    use_physics_features: bool      # default False
    use_geometry_features: bool     # default False
    use_node_type_features: bool    # default False
    predict_delta: bool             # default False
fluid:                       # read only by the momentum residual, and
  rho: float                 # only when the .mat does not carry them
  mu: float                  # one pair for every simulation...
  <simulation id>:           # ...or one pair per simulation
    rho: float
    mu: float
"""


from pathlib import Path

import yaml

from .data.features import FEATURE_SIZES
from .models import ARCHITECTURES


DEFAULTS = {
    "seed": 68,
    "dataset": {
        # The first snapshot is the artificial initial condition.
        "skip_initial": 1
    },
    "split": {
        "mode": "temporal",
        "train_fraction": 0.70,
        "val_fraction": 0.15,
        "gap": 1
    },
    "training": {
        "num_epochs": 200,
        "batch_size": 8,

        # Joint velocity-pressure training, the only mode left; the key
        # is kept for existing configs and must be true.
        "coupled": True,

        "velocity_data_weight": 1.0,
        "pressure_data_weight": 1.0,
        # "mse", or "mse_topk": MSE plus the MSE over the worst
        # topk_fraction of nodes.
        "data_loss": {
            "type": "mse",
            "topk_fraction": 0.10,
            "topk_weight": 1.0,
        },
        "lr_scheduler": {
            "type": "none",
            "step_size": 100,
            "gamma": 0.1,
        },

        "continuity_weight": 0.0,
        "momentum_weight": 0.0,
        # "fixed", or "dynamic": the two weights above are initial values,
        # raised by gradient ascent at dynamic_weight_learning_rate.
        "physics_weight_strategy": "fixed",
        "dynamic_weight_learning_rate": 1.0e-4,
        # Impose the Dirichlet BCs before the PDE residuals.
        "enforce_boundary_values": True,
        # Linearized FEM residual J(x*) (x - x*); off with weight 0.
        # type/topk_* as in data_loss, checked only when weight > 0.
        "newton_loss": {
            "weight": 0.0,
            "type": "mse",
            "topk_fraction": 0.10,
            "topk_weight": 1.0,
        },
    }
}

NETWORK_DEFAULTS = {
    "features": "time",
    "architecture": "gcn",
    "initialization": "default",
    "num_neurons": 64,
    "num_layers": 1,
    "dropout": 0.0,
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-5,

    # Extra input features. The input width and the feature vector are
    # derived from these flags, never from the network's name.
    "use_physics_features": False,
    "use_geometry_features": False,
    "use_node_type_features": False,

    # Predict the increment Y_target - X_input instead of the next state.
    "predict_delta": False,
}

# Network-level options of the removed non-coupled training, refused
# when they would change anything.
REMOVED_NETWORK_OPTIONS = {
    "use_predicted_velocity": "it is no longer supported",
    "continuity_weight": "use training.continuity_weight",
    "momentum_weight": "use training.momentum_weight",
}

MESHGRAPHNET_DEFAULTS = {
    "processor_size": 10,

    "num_layers_node_processor": 2,
    "num_layers_edge_processor": 2,

    "num_layers_node_encoder": 2,
    "num_layers_edge_encoder": 2,
    "num_layers_node_decoder": 2,

    "aggregation": "sum",
}


BSMS_DEFAULTS = {
    **MESHGRAPHNET_DEFAULTS,

    "num_mesh_levels": 2,
    "num_layers_bistride": 2,
    "bistride_unet_levels": 1,
}

def load_config(path):
    """Read, fill in defaults and validate an experiment file."""

    path = Path(path)

    with open(path) as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"{path} does not contain a YAML mapping.")

    config.setdefault("name", path.stem)
    config.setdefault("seed", DEFAULTS["seed"])

    for section in ("dataset", "split", "training"):
        merged = dict(DEFAULTS[section])
        merged.update(config.get(section) or {})
        config[section] = merged

    # Nested defaults for the data-loss configuration.
    data_loss = dict(
        DEFAULTS["training"]["data_loss"]
    )

    data_loss.update(
        config["training"].get("data_loss") or {}
    )

    config["training"]["data_loss"] = data_loss
    # Nested defaults for the learning-rate scheduler.
    lr_scheduler = dict(
        DEFAULTS["training"]["lr_scheduler"]
    )

    lr_scheduler.update(
        config["training"].get("lr_scheduler") or {}
    )

    config["training"]["lr_scheduler"] = lr_scheduler

    # Nested defaults for the Newton/FEM loss.
    newton_loss = dict(
        DEFAULTS["training"]["newton_loss"]
    )

    newton_loss.update(
        config["training"].get("newton_loss") or {}
    )

    config["training"]["newton_loss"] = newton_loss

    _validate(config, path)

    return config

def _validate(config, path):

    dataset = config.get("dataset", {})

    has_path = (
        "path" in dataset
        and dataset["path"] is not None
    )

    has_paths = (
        "paths" in dataset
        and isinstance(dataset["paths"], list)
        and len(dataset["paths"]) > 0
    )

    if not has_path and not has_paths:
        raise ValueError(
            f"{path}: dataset must contain either "
            "'dataset.path' for one simulation or "
            "'dataset.paths' for multiple simulations."
        )

    if has_path and has_paths:
        raise ValueError(
            f"{path}: use either 'dataset.path' or "
            "'dataset.paths', not both."
        )

    skip_initial = config["dataset"]["skip_initial"]

    if not isinstance(skip_initial, int) or skip_initial < 0:
        raise ValueError(
            f"{path}: dataset.skip_initial must be a non-negative "
            f"integer, got {skip_initial!r}."
        )

    networks = config.get("networks")

    if not networks:
        raise ValueError(f"{path}: at least one network is required.")

    training = config["training"]
    data_loss = training["data_loss"]

    if not isinstance(data_loss, dict):
        raise ValueError(
            f"{path}: training.data_loss must be a mapping."
        )

    data_loss_type = data_loss["type"]

    if data_loss_type not in {"mse", "mse_topk"}:
        raise ValueError(
            f"{path}: training.data_loss.type="
            f"{data_loss_type!r}; expected 'mse' or 'mse_topk'."
        )

    lr_scheduler = training["lr_scheduler"]

    if not isinstance(lr_scheduler, dict):
        raise ValueError(
            f"{path}: training.lr_scheduler must be a mapping."
        )

    scheduler_type = lr_scheduler["type"]

    if scheduler_type not in {"none", "step", "exponential"}:
        raise ValueError(
            f"{path}: training.lr_scheduler.type="
            f"{scheduler_type!r}; expected "
            "'none', 'step' or 'exponential'."
        )

    if scheduler_type == "step":

        step_size = lr_scheduler["step_size"]

        if not isinstance(step_size, int) or step_size < 1:
            raise ValueError(
                f"{path}: training.lr_scheduler.step_size="
                f"{step_size!r}; it must be an integer >= 1."
            )

    gamma = lr_scheduler["gamma"]

    if not isinstance(gamma, (int, float)):
        raise ValueError(
            f"{path}: training.lr_scheduler.gamma="
            f"{gamma!r}; it must be a number."
        )

    if not 0.0 < gamma <= 1.0:
        raise ValueError(
            f"{path}: training.lr_scheduler.gamma="
            f"{gamma!r}; expected 0 < gamma <= 1."
        )

    topk_fraction = data_loss["topk_fraction"]

    if not isinstance(topk_fraction, (int, float)):
        raise ValueError(
            f"{path}: training.data_loss.topk_fraction="
            f"{topk_fraction!r}; it must be a number."
        )

    if not 0.0 < topk_fraction <= 1.0:
        raise ValueError(
            f"{path}: training.data_loss.topk_fraction="
            f"{topk_fraction!r}; expected 0 < fraction <= 1."
        )

    topk_weight = data_loss["topk_weight"]

    if not isinstance(topk_weight, (int, float)):
        raise ValueError(
            f"{path}: training.data_loss.topk_weight="
            f"{topk_weight!r}; it must be a number."
        )

    if topk_weight < 0.0:
        raise ValueError(
            f"{path}: training.data_loss.topk_weight="
            f"{topk_weight!r}; it must not be negative."
        )

    newton_loss = training["newton_loss"]

    if not isinstance(newton_loss, dict):
        raise ValueError(
            f"{path}: training.newton_loss must be a mapping."
        )

    newton_weight = newton_loss["weight"]

    if not isinstance(newton_weight, (int, float)) or newton_weight < 0:
        raise ValueError(
            f"{path}: training.newton_loss.weight={newton_weight!r}; "
            "it must be a non-negative number."
        )

    if newton_weight > 0:

        if newton_loss["type"] not in {"mse", "mse_topk"}:
            raise ValueError(
                f"{path}: training.newton_loss.type="
                f"{newton_loss['type']!r}; expected 'mse' or 'mse_topk'."
            )

        newton_fraction = newton_loss["topk_fraction"]

        if (
            not isinstance(newton_fraction, (int, float))
            or not 0.0 < newton_fraction <= 1.0
        ):
            raise ValueError(
                f"{path}: training.newton_loss.topk_fraction="
                f"{newton_fraction!r}; expected 0 < fraction <= 1."
            )

        newton_topk_weight = newton_loss["topk_weight"]

        if (
            not isinstance(newton_topk_weight, (int, float))
            or newton_topk_weight < 0
        ):
            raise ValueError(
                f"{path}: training.newton_loss.topk_weight="
                f"{newton_topk_weight!r}; it must be a non-negative "
                "number."
            )

    if training["coupled"] is not True:
        raise ValueError(
            f"{path}: training.coupled={training['coupled']!r}; it must "
            "be true, the non-coupled training has been removed."
        )

    physics_weight_strategy = training[
        "physics_weight_strategy"
    ]

    if physics_weight_strategy not in {"fixed", "dynamic"}:
        raise ValueError(
            f"{path}: training.physics_weight_strategy="
            f"{physics_weight_strategy!r}; expected "
            "'fixed' or 'dynamic'."
        )
    for key in (
        "velocity_data_weight",
        "pressure_data_weight",
        "continuity_weight",
        "momentum_weight",
    ):

        weight = training[key]

        if not isinstance(weight, (int, float)):
            raise ValueError(
                f"{path}: training.{key}={weight!r}; "
                "it must be a number."
            )

        if weight < 0:
            raise ValueError(
                f"{path}: training.{key}={weight!r}; "
                "it must not be negative."
            )

    dynamic_weight_learning_rate = training[
        "dynamic_weight_learning_rate"
    ]

    if not isinstance(
        dynamic_weight_learning_rate,
        (int, float),
    ):
        raise ValueError(
            f"{path}: training.dynamic_weight_learning_rate="
            f"{dynamic_weight_learning_rate!r}; "
            "it must be a number."
        )

    if (
        physics_weight_strategy == "dynamic"
        and dynamic_weight_learning_rate <= 0.0
    ):
        raise ValueError(
            f"{path}: training.dynamic_weight_learning_rate="
            f"{dynamic_weight_learning_rate!r}; "
            "it must be positive when "
            "physics_weight_strategy='dynamic'."
        )
    if not isinstance(
        training["enforce_boundary_values"],
        bool,
    ):
        raise ValueError(
            f"{path}: training.enforce_boundary_values="
            f"{training['enforce_boundary_values']!r}; "
            "it must be true or false."
        )

    if set(networks) != {"velocity", "pressure"}:
        raise ValueError(
            f"{path}: the networks must be exactly 'velocity' and "
            f"'pressure', got {sorted(networks)}."
        )

    for name in ("velocity", "pressure"):

        if (networks[name] or {}).get("predicts") != name:
            raise ValueError(
                f"{path}: networks.{name}.predicts must be {name!r}."
            )

    mode = config["split"]["mode"]

    if mode not in ("temporal", "random", "simulation", "all_train"):
        raise ValueError(
            f"{path}: unknown split mode {mode!r}. "
            "Expected 'temporal', 'random', 'simulation' or 'all_train'."
        )
    # Whole simulations per block: at least three are needed.
    if mode == "simulation":

        if not has_paths:
            raise ValueError(
                f"{path}: split mode {mode!r} requires "
                "'dataset.paths' with multiple simulations."
            )

        if len(dataset["paths"]) < 3:
            raise ValueError(
                f"{path}: split mode {mode!r} keeps whole simulations "
                "together, so it needs at least 3 of them; "
                f"'dataset.paths' lists {len(dataset['paths'])}."
            )

    if mode == "simulation":

        train_fraction = config["split"].get("train_fraction")

        if train_fraction is None:
            raise ValueError(
                f"{path}: split mode 'simulation' requires "
                "'split.train_fraction'."
            )

        if not 0 < train_fraction < 1:
            raise ValueError(
                f"{path}: split.train_fraction must be between 0 and 1, "
                f"got {train_fraction!r}."
            )
    for name, network in networks.items():

        for key, hint in REMOVED_NETWORK_OPTIONS.items():

            if (network or {}).get(key):
                raise ValueError(
                    f"{path}: network {name!r} sets {key}="
                    f"{network[key]!r}; {hint}."
                )

        merged = dict(NETWORK_DEFAULTS)
        merged.update(network or {})

        if merged["architecture"] == "bsms":
            for key, value in BSMS_DEFAULTS.items():
                merged.setdefault(key, value)

        if merged["architecture"] == "meshgraphnet":
            for key, value in MESHGRAPHNET_DEFAULTS.items():
                merged.setdefault(key, value)

        networks[name] = merged

        if merged["features"] not in FEATURE_SIZES:
            raise ValueError(
                f"{path}: network {name!r} has "
                f"features={merged['features']!r}. "
                f"Expected one of {sorted(FEATURE_SIZES)}."
            )

        if merged["architecture"] not in ARCHITECTURES:
            raise ValueError(
                f"{path}: network {name!r} has "
                f"architecture={merged['architecture']!r}. "
                f"Expected one of {sorted(ARCHITECTURES)}."
            )

        # The training loop runs the pressure network with the BSMS
        # hierarchy and the velocity network on the plain graph.
        if merged["initialization"] not in {
            "default",
            "zero",
            "zero_output",
        }:
            raise ValueError(
                f"{path}: network {name!r} has "
                f"initialization={merged['initialization']!r}. "
                "Expected 'default', 'zero' or 'zero_output'."
            )

        if merged["architecture"] == "bsms":

            for key in (
                "num_mesh_levels",
                "num_layers_bistride",
                "bistride_unet_levels",
            ):
                value = merged[key]

                if not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"{path}: BSMS network {name!r} requires "
                        f"{key} to be an integer >= 1, got {value!r}."
                    )

        if merged["architecture"] in {"meshgraphnet", "bsms"}:

            for key in (
                "processor_size",
                "num_layers_node_processor",
                "num_layers_edge_processor",
                "num_layers_node_encoder",
                "num_layers_edge_encoder",
                "num_layers_node_decoder",
            ):
                value = merged[key]

                if not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"{path}: NVIDIA graph network {name!r} requires "
                        f"{key} to be an integer >= 1, got {value!r}."
                    )

            if merged["aggregation"] not in {"sum", "mean"}:
                raise ValueError(
                    f"{path}: network {name!r} has "
                    f"aggregation={merged['aggregation']!r}; "
                    "expected 'sum' or 'mean'."
                )

        for flag in (
            "use_physics_features",
            "use_geometry_features",
            "use_node_type_features",
            "predict_delta",
        ):

            if not isinstance(merged[flag], bool):
                raise ValueError(
                    f"{path}: network {name!r} has "
                    f"{flag}={merged[flag]!r}; it must be true or false."
                )


