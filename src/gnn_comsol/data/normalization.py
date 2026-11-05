"""
Normalization utilities for state, physics, geometry and timestep data.

State convention:
    [u, v, p]

Physical nondimensionalization:
    u*  = u / U_inf
    v*  = v / U_inf
    p*  = p / (rho * U_inf**2)
    dt* = dt * U_inf / D

where:
    D = 2R

Physics and geometry normalization statistics are computed
from training data only.
"""

import numpy as np
import torch


# State layout

VELOCITY_COLUMNS = slice(0, 2)
PRESSURE_COLUMNS = slice(2, 3)
STATE_COLUMNS = slice(0, 3)

VARIABLE_NAMES = ["u", "v", "p"]

TARGET_COLUMNS = {
    "velocity": VELOCITY_COLUMNS,
    "pressure": PRESSURE_COLUMNS,
    "state": STATE_COLUMNS,
}


# Extra features

PHYSICS_FEATURE_NAMES = [
    "du_dx",
    "du_dy",
    "dv_dx",
    "dv_dy",
    "div_conv",
]

GEOMETRY_FEATURE_NAMES = [
    "wall_geom_x",
    "wall_geom_y",
    "inlet_geom_x",
    "inlet_geom_y",
    "outlet_geom_x",
    "outlet_geom_y",
]

NUM_PHYSICS_FEATURES = len(PHYSICS_FEATURE_NAMES)
NUM_GEOMETRY_FEATURES = len(GEOMETRY_FEATURE_NAMES)


# Helpers

def _require_training_simulations(train_simulations):
    if not train_simulations:
        raise ValueError("At least one training simulation is required.")


def _simulation_label(simulation):
    return (
        f"Simulation {simulation.simulation_id} "
        f"({simulation.file_path})"
    )


def _compute_feature_statistics(
    train_simulations,
    attribute,
    num_features,
    expected_ndim,
):
    """Mean/std of a feature array over all training simulations."""

    _require_training_simulations(train_simulations)

    arrays = []

    for simulation in train_simulations:

        features = getattr(simulation, attribute)

        if features is None:
            raise ValueError(
                f"{_simulation_label(simulation)} has no {attribute}."
            )

        features = np.asarray(features, dtype=np.float64)

        if features.ndim != expected_ndim:
            raise ValueError(
                f"{_simulation_label(simulation)}: {attribute} must have "
                f"{expected_ndim} dimensions, got shape {features.shape}."
            )

        if features.shape[-1] != num_features:
            raise ValueError(
                f"{_simulation_label(simulation)}: expected "
                f"{num_features} {attribute}, got {features.shape[-1]}."
            )

        if not np.all(np.isfinite(features)):
            raise ValueError(
                f"{_simulation_label(simulation)}: "
                f"{attribute} contains NaN or Inf."
            )

        arrays.append(
            features.reshape(-1, num_features)
        )

    values = np.concatenate(arrays, axis=0)

    return (
        values.mean(axis=0),
        values.std(axis=0) + 1e-8,
    )


def _state_scale(simulation):
    """Physical scale [U_inf, U_inf, rho * U_inf^2]."""

    if simulation.U_inf is None:
        raise ValueError(
            f"Simulation {simulation.simulation_id} has no U_inf."
        )

    if simulation.rho is None:
        raise ValueError(
            f"Simulation {simulation.simulation_id} has no rho."
        )

    U_inf = float(simulation.U_inf)
    rho = float(simulation.rho)

    if U_inf <= 0:
        raise ValueError(f"U_inf must be positive, got {U_inf}.")

    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}.")

    return np.array(
        [U_inf, U_inf, rho * U_inf**2],
        dtype=np.float64,
    )
def _timestep_scale(simulation):
    """
    Convective time scale L_ref / U_inf.

    Here L_ref is the cylinder diameter D = 2R.
    """

    if simulation.U_inf is None:
        raise ValueError(
            f"Simulation {simulation.simulation_id} has no U_inf."
        )

    if simulation.R is None:
        raise ValueError(
            f"Simulation {simulation.simulation_id} has no cylinder radius R."
        )

    U_inf = float(simulation.U_inf)
    R = float(simulation.R)

    if U_inf <= 0:
        raise ValueError(f"U_inf must be positive, got {U_inf}.")

    if R <= 0:
        raise ValueError(f"R must be positive, got {R}.")

    D = 2.0 * R

    return D / U_inf

# Feature statistics

def compute_multi_simulation_physics_normalization_parameters(
    train_simulations,
):
    """Physics-feature mean/std over training simulations."""

    return _compute_feature_statistics(
        train_simulations,
        attribute="physics_features",
        num_features=NUM_PHYSICS_FEATURES,
        expected_ndim=3,
    )


def compute_multi_simulation_geometry_normalization_parameters(
    train_simulations,
):
    """Geometry-feature mean/std over training simulations."""

    return _compute_feature_statistics(
        train_simulations,
        attribute="geometry_features",
        num_features=NUM_GEOMETRY_FEATURES,
        expected_ndim=2,
    )

def compute_timestep_statistics(train_simulations):
    """Mean and standard deviation of dt over training simulations."""
    _require_training_simulations(train_simulations)

    delta_t = np.concatenate(
        [
            np.asarray(
                simulation.delta_t,
                dtype=np.float64,
            )
            for simulation in train_simulations
        ]
    )

    dt_mean = float(delta_t.mean())
    dt_std = float(delta_t.std() + 1e-8)

    return dt_mean, dt_std

# Normalizers

class _BaseNormalizer:
    """Common implementation for independent feature standardization."""

    KIND = None
    NUM_FEATURES = None
    FEATURE_NAMES = None

    def __init__(self, mean, std):

        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)

        if self.mean.shape != self.std.shape:
            raise ValueError(
                f"mean {self.mean.shape} and std {self.std.shape} "
                "must have the same shape."
            )

        if (
            self.NUM_FEATURES is not None
            and self.mean.shape != (self.NUM_FEATURES,)
        ):
            raise ValueError(
                f"Expected {self.NUM_FEATURES} normalization "
                f"parameters, got {self.mean.shape}."
            )

        if np.any(self.std <= 0):
            raise ValueError(
                f"std must be strictly positive, got {self.std}."
            )

    @staticmethod
    def _as_like(parameters, values):

        if isinstance(values, torch.Tensor):
            return torch.as_tensor(
                parameters,
                dtype=values.dtype,
                device=values.device,
            )

        return parameters

    def transform(self, values):
        return (
            values - self._as_like(self.mean, values)
        ) / self._as_like(self.std, values)

    def inverse_transform(self, values):
        return (
            values * self._as_like(self.std, values)
            + self._as_like(self.mean, values)
        )

    def to_dict(self):

        state = {
            "kind": self.KIND,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

        if self.FEATURE_NAMES is not None:
            state["feature_names"] = self.FEATURE_NAMES

        return state

    @classmethod
    def from_dict(cls, state):

        kind = state.get("kind")

        if kind != cls.KIND:
            raise ValueError(
                f"Unknown normalization kind {kind!r}; "
                f"expected {cls.KIND!r}."
            )

        return cls(
            state["mean"],
            state["std"],
        )

    def __repr__(self):

        return (
            f"{self.__class__.__name__}("
            f"kind='{self.KIND}', "
            f"mean={np.array2string(self.mean, precision=4)}, "
            f"std={np.array2string(self.std, precision=4)})"
        )

class PhysicsNormalizer(_BaseNormalizer):
    """Standardization of physics-derived node features."""

    KIND = "physics_standardize"
    NUM_FEATURES = NUM_PHYSICS_FEATURES
    FEATURE_NAMES = PHYSICS_FEATURE_NAMES


class GeometryNormalizer(_BaseNormalizer):
    """Standardization of static geometry features."""

    KIND = "geometry_standardize"
    NUM_FEATURES = NUM_GEOMETRY_FEATURES
    FEATURE_NAMES = GEOMETRY_FEATURE_NAMES


# Complete simulation normalization

def normalize_simulation(
    simulation,
    dt_mean,
    physics_normalizer=None,
    geometry_normalizer=None,
    predict_delta=False,
):
    """
    Normalize one simulation while preserving physical metadata.

    State and delta use physical nondimensionalization.
    Physics/geometry features use their fitted standardizers.
    Boundary conditions and FEM Jacobians remain in physical units.
    """

    state_scale = _state_scale(simulation)
    time_scale = _timestep_scale(simulation)

    normalized = {
        "X": simulation.X_input / state_scale,
        "Y": simulation.Y_target / state_scale,
        "dt": simulation.delta_t / time_scale,
        "dt_physical": simulation.delta_t,

        "edge_index": simulation.edge_index,
        "edge_weight": simulation.edge_weight,
        "pos": simulation.pos,

        "wall_mask": simulation.wall_mask,
        "inlet_mask": simulation.inlet_mask,
        "outlet_mask": simulation.outlet_mask,

        "velocity_bc_target": simulation.velocity_bc_target,
        "velocity_dirichlet_mask": simulation.velocity_dirichlet_mask,
        "pressure_bc_target": simulation.pressure_bc_target,
        "pressure_dirichlet_mask": simulation.pressure_dirichlet_mask,

        "neighbors": simulation.neighbors,
        "G_wlsq": simulation.G_wlsq,
        "cell_index": simulation.cell_index,

        "simulation_id": simulation.simulation_id,
        "file_path": simulation.file_path,

        # COMSOL FEM Jacobian: already aligned in loading.py and
        # intentionally kept in physical FEM units.
        "jacobian_shared_pattern": simulation.jacobian_shared_pattern,
        "jacobian_row": simulation.jacobian_row,
        "jacobian_col": simulation.jacobian_col,
        "jacobian_values": simulation.jacobian_values,
        "jacobian_shape": simulation.jacobian_shape,
        "jacobian_dof_p": simulation.jacobian_dof_p,
        "jacobian_dof_u": simulation.jacobian_dof_u,
        "jacobian_dof_v": simulation.jacobian_dof_v,
        "jacobian_target_time": simulation.jacobian_target_time,
    }

    if physics_normalizer is not None:

        if simulation.physics_features is None:
            raise ValueError(
                f"{_simulation_label(simulation)} has no physics_features."
            )

        normalized["physics_features"] = (
            physics_normalizer.transform(
                simulation.physics_features
            )
        )

    if geometry_normalizer is not None:

        if simulation.geometry_features is None:
            raise ValueError(
                f"{_simulation_label(simulation)} has no geometry_features."
            )

        normalized["geometry_features"] = (
            geometry_normalizer.transform(
                simulation.geometry_features
            )
        )

    if predict_delta:
        normalized["delta"] = (
            simulation.Y_target - simulation.X_input
        ) / state_scale

    return normalized


def uses_physics_features(config):
    """Whether at least one network uses physics-derived features."""

    return any(
        network["use_physics_features"]
        for network in config["networks"].values()
    )


def uses_geometry_features(config):
    """Whether at least one network uses geometry-derived features."""

    return any(
        network["use_geometry_features"]
        for network in config["networks"].values()
    )


def uses_delta_prediction(config):
    """Whether at least one network predicts state increments."""

    return any(
        network["predict_delta"]
        for network in config["networks"].values()
    )

def build_normalizers(train_simulations, config):
    """
    Fit feature normalizers and timestep statistics using training data only.
    """

    dt_mean, dt_std = compute_timestep_statistics(
        train_simulations
    )

    physics_normalizer = None

    if uses_physics_features(config):

        physics_mean, physics_std = (
            compute_multi_simulation_physics_normalization_parameters(
                train_simulations
            )
        )

        physics_normalizer = PhysicsNormalizer(
            physics_mean,
            physics_std,
        )

    geometry_normalizer = None

    if uses_geometry_features(config):

        geometry_mean, geometry_std = (
            compute_multi_simulation_geometry_normalization_parameters(
                train_simulations
            )
        )

        geometry_normalizer = GeometryNormalizer(
            geometry_mean,
            geometry_std,
        )

    return (
        physics_normalizer,
        geometry_normalizer,
        dt_mean,
        dt_std,
    )


def normalize_splits(
    splits,
    physics_normalizer,
    geometry_normalizer,
    dt_mean,
    config,
):
    """Normalize the train, validation and test simulations."""

    predict_delta = uses_delta_prediction(config)

    blocks = {
        "train": splits.train,
        "val": splits.val,
        "test": splits.test,
    }

    return {
        split_name: [
            normalize_simulation(
                simulation,
                dt_mean,
                physics_normalizer=physics_normalizer,
                geometry_normalizer=geometry_normalizer,
                predict_delta=predict_delta,
            )
            for simulation in split_simulations
        ]
        for split_name, split_simulations in blocks.items()
    }

def build_state_scales(simulations):
    """Physical state scale of every simulation."""

    return {
        simulation.simulation_id: _state_scale(simulation)
        for simulation in simulations
    }