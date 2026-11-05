"""
The scaling contract: the state is divided by its physical scale, the
features are standardized, and every transform has an exact inverse.
"""

import numpy as np
import pytest

from gnn_comsol.data.loading import RawDataset
from gnn_comsol.data.normalization import (
    NUM_PHYSICS_FEATURES,
    PHYSICS_FEATURE_NAMES,
    PhysicsNormalizer,

    compute_multi_simulation_physics_normalization_parameters,
    compute_timestep_statistics,
    normalize_simulation,
)


def state_simulation(simulation_id=0, num_samples=20, num_nodes=50,
                     U_inf=1.5, rho=1.2, seed=0):
    """A channel-shaped state: u ~ U_inf, v ~ 0, p of order rho U_inf^2."""

    rng = np.random.default_rng(seed)

    u = U_inf + 0.15 * rng.normal(size=(num_samples, num_nodes))
    v = 0.02 * rng.normal(size=(num_samples, num_nodes))
    p = 2000.0 + 300.0 * rng.normal(size=(num_samples, num_nodes))

    X = np.stack([u, v, p], axis=-1)

    return RawDataset(
        X_input=X,
        Y_target=1.1 * X,
        edge_index=np.zeros((2, 1), dtype=int),
        edge_weight=np.ones(1),
        delta_t=np.linspace(0.01, 0.05, num_samples),
        pos=np.zeros((num_nodes, 2)),
        simulation_id=simulation_id,
        U_inf=U_inf,
        rho=rho,
    )


def test_state_is_divided_by_its_physical_scale():
    """[u, v, p] / [U_inf, U_inf, rho U_inf^2], and nothing else."""

    simulation = state_simulation()

    normalized = normalize_simulation(
        simulation, dt_mean=0.03, predict_delta=True
    )

    scale = np.array([1.5, 1.5, 1.2 * 1.5**2])

    assert np.allclose(normalized["X"] * scale, simulation.X_input)
    assert np.allclose(normalized["Y"] * scale, simulation.Y_target)
    assert np.allclose(
        normalized["delta"] * scale,
        simulation.Y_target - simulation.X_input,
    )
    assert np.allclose(normalized["dt"], simulation.delta_t / 0.03)
    assert np.array_equal(normalized["dt_physical"], simulation.delta_t)


def test_u_and_v_share_one_scale():
    """
    Different scales for u and v would stretch the velocity field along
    one axis and the divergence would no longer be proportional to the
    physical one.
    """

    simulation = state_simulation()

    normalized = normalize_simulation(simulation, dt_mean=1.0)

    for column in (0, 1):

        physical = np.gradient(simulation.X_input[:, :, column], axis=1)
        scaled = np.gradient(normalized["X"][:, :, column], axis=1)

        assert np.allclose(scaled * 1.5, physical)


@pytest.mark.parametrize("field", ["U_inf", "rho"])
def test_a_simulation_without_its_physical_scale_is_refused(field):

    simulation = state_simulation(simulation_id=7)
    setattr(simulation, field, None)

    with pytest.raises(ValueError, match="Simulation 7"):
        normalize_simulation(simulation, dt_mean=1.0)


def test_timestep_statistics_pool_every_training_transition():

    simulations = [
        state_simulation(0, num_samples=10),
        state_simulation(1, num_samples=25),
    ]

    dt_mean, dt_std = compute_timestep_statistics(simulations)

    pooled = np.concatenate([s.delta_t for s in simulations])

    assert dt_mean == pytest.approx(pooled.mean())
    assert dt_std == pytest.approx(pooled.std() + 1e-8)


def test_degenerate_std_is_refused():

    std = np.ones(NUM_PHYSICS_FEATURES)
    std[2] = 0.0

    with pytest.raises(ValueError):
        PhysicsNormalizer(np.zeros(NUM_PHYSICS_FEATURES), std)


# ---------------------------------------------------------------------
# Physics-derived features
# ---------------------------------------------------------------------

def physics_simulation(simulation_id, num_samples, num_nodes, scales,
                       seed=0):
    """A RawDataset carrying only what the physics statistics need."""

    rng = np.random.default_rng(seed)

    features = rng.normal(
        size=(num_samples, num_nodes, NUM_PHYSICS_FEATURES)
    ) * scales

    return RawDataset(
        X_input=np.zeros((num_samples, num_nodes, 3)),
        Y_target=np.zeros((num_samples, num_nodes, 3)),
        edge_index=np.zeros((2, 1), dtype=int),
        edge_weight=np.ones(1),
        delta_t=np.ones(num_samples),
        pos=np.zeros((num_nodes, 2)),
        physics_features=features,
        simulation_id=simulation_id
    )


def test_physics_parameters_pool_meshes_of_different_sizes():
    """
    Every node at every timestep counts once, whatever mesh it is on.

    Simulations have different node counts, so the statistics have to be
    taken over the flattened sample-node axis rather than per simulation
    and then averaged.
    """

    scales = np.array([1.0, 2.0, 0.5, 4.0, 100.0])

    simulations = [
        physics_simulation(0, 12, 7, scales, seed=1),
        physics_simulation(1, 20, 13, scales, seed=2),
        physics_simulation(2, 5, 30, scales, seed=3)
    ]

    mean, std = (
        compute_multi_simulation_physics_normalization_parameters(
            simulations
        )
    )

    assert mean.shape == (NUM_PHYSICS_FEATURES,)
    assert std.shape == (NUM_PHYSICS_FEATURES,)

    # Same answer as pooling the nodes by hand
    pooled = np.concatenate(
        [
            simulation.physics_features.reshape(
                -1, NUM_PHYSICS_FEATURES
            )
            for simulation in simulations
        ]
    )

    assert np.allclose(mean, pooled.mean(axis=0))
    assert np.allclose(std, pooled.std(axis=0) + 1e-8)

    # The columns really are on different scales, so a normalizer that
    # did nothing would not pass the round trip below.
    assert std.max() / std.min() > 10


def test_physics_normalizer_round_trips_and_whitens():

    scales = np.array([1.0, 2.0, 0.5, 4.0, 100.0])

    simulations = [physics_simulation(0, 30, 11, scales, seed=4)]

    mean, std = (
        compute_multi_simulation_physics_normalization_parameters(
            simulations
        )
    )

    physics_normalizer = PhysicsNormalizer(mean, std)

    features = simulations[0].physics_features

    normalized = physics_normalizer.transform(features)

    assert np.allclose(normalized.mean(axis=(0, 1)), 0.0, atol=1e-8)
    assert np.allclose(normalized.std(axis=(0, 1)), 1.0, atol=1e-6)

    assert np.allclose(
        physics_normalizer.inverse_transform(normalized), features
    )


def test_physics_normalizer_serializes_with_its_feature_names():
    """
    A stored normalizer must be readable back without knowing which
    features it was built for, and must refuse a foreign convention.
    """

    physics_normalizer = PhysicsNormalizer(
        np.zeros(NUM_PHYSICS_FEATURES), np.ones(NUM_PHYSICS_FEATURES)
    )

    stored = physics_normalizer.to_dict()

    assert stored["feature_names"] == PHYSICS_FEATURE_NAMES

    restored = PhysicsNormalizer.from_dict(stored)

    assert np.allclose(restored.mean, physics_normalizer.mean)
    assert np.allclose(restored.std, physics_normalizer.std)

    with pytest.raises(ValueError, match="physics normalization"):
        PhysicsNormalizer.from_dict({**stored, "kind": "standardize"})


def test_physics_parameters_name_the_simulation_that_lacks_them():

    scales = np.ones(NUM_PHYSICS_FEATURES)

    good = physics_simulation(0, 5, 4, scales)

    bad = physics_simulation(1, 5, 4, scales)
    bad.physics_features = None

    with pytest.raises(ValueError, match="Simulation 1"):
        compute_multi_simulation_physics_normalization_parameters(
            [good, bad]
        )

