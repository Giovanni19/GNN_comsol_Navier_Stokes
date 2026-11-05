"""Experiment files: defaults and validation."""

import pytest
import yaml

from gnn_comsol.config import load_config


BASE = {
    "name": "unit_test",
    "dataset": {"path": "data/whatever.mat"},
    "networks": {
        "velocity": {
            "predicts": "velocity",
            "features": "time",
            "architecture": "gcn"
        },
        "pressure": {
            "predicts": "pressure",
            "features": "time_fourier",
            "architecture": "bsms"
        }
    }
}


def write(tmp_path, config, name="experiment.yaml"):

    path = tmp_path / name

    with open(path, "w") as handle:
        yaml.safe_dump(config, handle)

    return path


def test_defaults_are_filled_in(tmp_path):

    config = load_config(write(tmp_path, BASE))

    assert config["seed"] == 68
    assert config["split"]["mode"] == "temporal"
    assert config["split"]["gap"] == 1
    assert config["training"]["num_epochs"] == 200

    # Network defaults too
    assert config["networks"]["velocity"]["num_neurons"] == 64


def test_explicit_values_win_over_defaults(tmp_path):

    custom = {**BASE, "split": {"mode": "random", "gap": 0}}

    config = load_config(write(tmp_path, custom))

    assert config["split"]["mode"] == "random"
    assert config["split"]["gap"] == 0
    # untouched keys still get their default
    assert config["split"]["train_fraction"] == 0.70


def test_shipped_configs_are_valid():
    """The configs in configs/ must load."""

    from pathlib import Path

    configs = Path(__file__).resolve().parents[1] / "configs"

    files = sorted(configs.glob("*.yaml"))

    assert files, "no config files found"

    for path in files:
        load_config(path)


def test_a_missing_network_is_rejected(tmp_path):
    """Training needs both networks."""

    partial = {
        **BASE,
        "networks": {"velocity": {"predicts": "velocity"}}
    }

    with pytest.raises(ValueError, match="must be exactly"):
        load_config(write(tmp_path, partial))


def test_other_network_names_are_rejected(tmp_path):

    overlapping = {
        **BASE,
        "networks": {
            "a": {"predicts": "state"},
            "b": {"predicts": "pressure"}
        }
    }

    with pytest.raises(ValueError, match="must be exactly"):
        load_config(write(tmp_path, overlapping))


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("predicts", "temperature", "predicts"),
        ("features", "wavelet", "features"),
        ("architecture", "transformer", "architecture"),
    ],
)
def test_unknown_network_options_are_rejected(
    tmp_path, field, value, message
):

    broken = {
        **BASE,
        "networks": {
            "velocity": {"predicts": "velocity"},
            "pressure": {"predicts": "pressure", field: value}
        }
    }

    with pytest.raises(ValueError, match=message):
        load_config(write(tmp_path, broken))


def test_missing_dataset_path_is_rejected(tmp_path):

    broken = {k: v for k, v in BASE.items() if k != "dataset"}

    with pytest.raises(ValueError, match="dataset.path"):
        load_config(write(tmp_path, broken))


def test_unknown_split_mode_is_rejected(tmp_path):

    broken = {**BASE, "split": {"mode": "kfold"}}

    with pytest.raises(ValueError, match="split mode"):
        load_config(write(tmp_path, broken))


# ---------------------------------------------------------------------
# One simulation or many
# ---------------------------------------------------------------------

MULTI = {
    "name": "multi",
    "dataset": {"paths": ["a.mat", "b.mat", "c.mat"]},
    "split": {"mode": "simulation", "train_fraction": 0.70},
    "networks": BASE["networks"]
}


def test_paths_is_accepted_for_multiple_simulations(tmp_path):

    config = load_config(write(tmp_path, MULTI))

    assert len(config["dataset"]["paths"]) == 3


def test_path_and_paths_together_are_rejected(tmp_path):

    both = {
        **MULTI,
        "dataset": {"path": "a.mat", "paths": ["a.mat", "b.mat", "c.mat"]}
    }

    with pytest.raises(ValueError, match="not both"):
        load_config(write(tmp_path, both))


def test_simulation_split_requires_paths(tmp_path):

    single = {
        **BASE,
        "split": {"mode": "simulation", "train_fraction": 0.70}
    }

    with pytest.raises(ValueError, match="requires"):
        load_config(write(tmp_path, single))


@pytest.mark.parametrize("fraction", [0.0, 1.0, 1.5, -0.2])
def test_simulation_split_rejects_impossible_fractions(
    tmp_path, fraction
):

    bad = {
        **MULTI,
        "split": {"mode": "simulation", "train_fraction": fraction}
    }

    with pytest.raises(ValueError, match="between 0 and 1"):
        load_config(write(tmp_path, bad))


# ---------------------------------------------------------------------
# Extra input features
# ---------------------------------------------------------------------

BSMS_PRESSURE = {
    "predicts": "pressure",
    "features": "time",
    "architecture": "bsms",
}


def test_extra_feature_flags_default_to_off(tmp_path):
    """
    The width of a model must follow the configuration, so the flags
    have to exist with a known value rather than being read with .get()
    in whichever module happens to need them.
    """

    config = load_config(write(tmp_path, BASE))

    for network in config["networks"].values():
        assert network["use_physics_features"] is False
        assert network["use_geometry_features"] is False
        assert network["use_node_type_features"] is False
        assert network["predict_delta"] is False


@pytest.mark.parametrize(
    "flag",
    ["use_physics_features", "use_geometry_features", "use_node_type_features"]
)
def test_extra_feature_flags_must_be_boolean(tmp_path, flag):

    broken = {
        **BASE,
        "networks": {
            "velocity": {"predicts": "velocity"},
            "pressure": {**BSMS_PRESSURE, flag: "yes"}
        }
    }

    with pytest.raises(ValueError, match="true or false"):
        load_config(write(tmp_path, broken))


def test_extra_features_are_accepted_on_bsms(tmp_path):

    accepted = {
        **BASE,
        "networks": {
            "velocity": {"predicts": "velocity"},
            "pressure": {
                **BSMS_PRESSURE,
                "use_physics_features": True,
                "use_geometry_features": True,
                "use_node_type_features": True
            }
        }
    }

    config = load_config(write(tmp_path, accepted))

    pressure = config["networks"]["pressure"]

    assert pressure["use_physics_features"] is True
    assert pressure["use_geometry_features"] is True
    assert pressure["use_node_type_features"] is True


@pytest.mark.parametrize("skip", [-1, 1.5, "two"])
def test_invalid_skip_initial_is_rejected(tmp_path, skip):

    bad = {**BASE, "dataset": {"path": "a.mat", "skip_initial": skip}}

    with pytest.raises(ValueError, match="skip_initial"):
        load_config(write(tmp_path, bad))


# ---------------------------------------------------------------------
# predict_delta
# ---------------------------------------------------------------------

def test_predict_delta_works_on_any_architecture(tmp_path):
    """
    predict_delta only changes which target array the loader builds
    and how the prediction is reconstructed afterwards.
    """

    accepted = {
        **BASE,
        "networks": {
            "velocity": {**BASE["networks"]["velocity"], "predict_delta": True},
            "pressure": {**BASE["networks"]["pressure"], "predict_delta": True}
        }
    }

    config = load_config(write(tmp_path, accepted))

    assert config["networks"]["velocity"]["predict_delta"] is True
    assert config["networks"]["pressure"]["predict_delta"] is True


def test_predict_delta_must_be_boolean(tmp_path):

    broken = {
        **BASE,
        "networks": {
            "velocity": {**BASE["networks"]["velocity"], "predict_delta": "yes"},
            "pressure": BASE["networks"]["pressure"]
        }
    }

    with pytest.raises(ValueError, match="true or false"):
        load_config(write(tmp_path, broken))


def test_newton_loss_defaults_are_filled_in(tmp_path):

    newton = load_config(write(tmp_path, BASE))["training"]["newton_loss"]

    assert newton == {
        "weight": 0.0,
        "type": "mse",
        "topk_fraction": 0.10,
        "topk_weight": 1.0,
    }


def test_newton_loss_settings_are_kept(tmp_path):
    """type and top-k used to be dropped: only the weight was read."""

    custom = {
        **BASE,
        "training": {
            "newton_loss": {
                "weight": 1.0,
                "type": "mse_topk",
                "topk_fraction": 0.2,
            }
        },
    }

    newton = load_config(write(tmp_path, custom))["training"]["newton_loss"]

    assert newton["type"] == "mse_topk"
    assert newton["topk_fraction"] == 0.2
    assert newton["topk_weight"] == 1.0


@pytest.mark.parametrize(
    "settings",
    [
        {"weight": -1.0},
        {"weight": 1.0, "type": "l1"},
        {"weight": 1.0, "topk_fraction": 0.0},
        {"weight": 1.0, "topk_weight": -1.0},
    ],
)
def test_invalid_newton_loss_is_rejected(tmp_path, settings):

    custom = {**BASE, "training": {"newton_loss": settings}}

    with pytest.raises(ValueError, match="newton_loss"):
        load_config(write(tmp_path, custom))


def test_disabled_newton_loss_settings_are_not_checked(tmp_path):

    custom = {
        **BASE,
        "training": {
            "newton_loss": {"weight": 0.0, "topk_fraction": 0.0}
        },
    }

    load_config(write(tmp_path, custom))


# ---------------------------------------------------------------------
# Coupled training only
# ---------------------------------------------------------------------

def test_coupled_defaults_to_true(tmp_path):

    assert load_config(write(tmp_path, BASE))["training"]["coupled"] is True


def test_non_coupled_training_is_refused(tmp_path):

    custom = {**BASE, "training": {"coupled": False}}

    with pytest.raises(ValueError, match="non-coupled"):
        load_config(write(tmp_path, custom))


@pytest.mark.parametrize(
    "velocity_arch,pressure_arch",
    [("gcn", "gcn_virtual_node"), ("bsms", "bsms")],
)
def test_only_the_pressure_network_is_bsms(
    tmp_path, velocity_arch, pressure_arch
):

    custom = {
        **BASE,
        "networks": {
            "velocity": {"predicts": "velocity", "architecture": velocity_arch},
            "pressure": {"predicts": "pressure", "architecture": pressure_arch},
        },
    }

    with pytest.raises(ValueError, match="bsms"):
        load_config(write(tmp_path, custom))


@pytest.mark.parametrize(
    "key,value,hint",
    [
        ("use_predicted_velocity", True, "no longer supported"),
        ("continuity_weight", 1.0e-3, "training.continuity_weight"),
        ("momentum_weight", -1.0, "training.momentum_weight"),
    ],
)
def test_removed_network_options_are_refused(tmp_path, key, value, hint):

    custom = {
        **BASE,
        "networks": {
            **BASE["networks"],
            "velocity": {**BASE["networks"]["velocity"], key: value},
        },
    }

    with pytest.raises(ValueError, match=hint):
        load_config(write(tmp_path, custom))


def test_removed_network_options_left_off_are_accepted(tmp_path):
    """Older configs write them with their old default."""

    custom = {
        **BASE,
        "networks": {
            **BASE["networks"],
            "pressure": {
                **BASE["networks"]["pressure"],
                "use_predicted_velocity": False,
                "continuity_weight": 0.0,
            },
        },
    }

    load_config(write(tmp_path, custom))
