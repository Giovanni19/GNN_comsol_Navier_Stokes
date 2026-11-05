"""
The node feature vector.

Its layout is a property of the trained weights: the training loaders
and the evaluation script both build it with build_features, and must
get the same blocks in the same order.
"""

import numpy as np
import pytest

from gnn_comsol.data.features import (
    FEATURE_SIZES,
    NUM_NODE_TYPE_FEATURES,
    build_features,
    build_node_type_features,
    features_size,
)
from gnn_comsol.data.normalization import (
    NUM_GEOMETRY_FEATURES,
    NUM_PHYSICS_FEATURES,
)


SAMPLES, NODES = 6, 5


@pytest.fixture
def inputs():

    rng = np.random.default_rng(0)

    return {
        "X": rng.normal(size=(SAMPLES, NODES, 3)),
        "dt": rng.normal(size=SAMPLES),
        "physics": rng.normal(
            size=(SAMPLES, NODES, NUM_PHYSICS_FEATURES)
        ),
        "geometry": rng.normal(size=(NODES, NUM_GEOMETRY_FEATURES)),
        "node_type": build_node_type_features(
            wall_mask=[1, 0, 0, 0, 1],
            inlet_mask=[1, 1, 0, 0, 0],
            outlet_mask=[0, 0, 0, 1, 0],
        ),
    }


@pytest.mark.parametrize("encoding", sorted(FEATURE_SIZES))
@pytest.mark.parametrize("physics", [False, True])
@pytest.mark.parametrize("geometry", [False, True])
@pytest.mark.parametrize("node_type", [False, True])
def test_size_matches_what_the_builder_produces(
    inputs, encoding, physics, geometry, node_type
):
    """
    The number the model is sized with is the number of columns the
    loader hands it.
    """

    features = build_features(
        encoding,
        inputs["X"],
        inputs["dt"],
        physics_features=inputs["physics"] if physics else None,
        geometry_features=inputs["geometry"] if geometry else None,
        node_type_features=inputs["node_type"] if node_type else None,
    )

    expected = features_size(
        encoding,
        use_physics_features=physics,
        use_geometry_features=geometry,
        use_node_type_features=node_type,
    )

    assert features.shape == (SAMPLES, NODES, expected)


def test_blocks_keep_their_documented_order(inputs):
    """
    [ time encoding | physics | geometry | node type ]

    Getting this wrong does not change any shape, so nothing else would
    catch it.
    """

    features = build_features(
        "time",
        inputs["X"],
        inputs["dt"],
        physics_features=inputs["physics"],
        geometry_features=inputs["geometry"],
        node_type_features=inputs["node_type"],
    )

    base_end = FEATURE_SIZES["time"]
    physics_end = base_end + NUM_PHYSICS_FEATURES
    geometry_end = physics_end + NUM_GEOMETRY_FEATURES

    assert np.allclose(
        features[..., :base_end],
        build_features("time", inputs["X"], inputs["dt"]),
    )
    assert np.allclose(
        features[..., base_end:physics_end], inputs["physics"]
    )
    assert np.allclose(
        features[..., physics_end:geometry_end], inputs["geometry"]
    )
    assert np.allclose(
        features[..., geometry_end:], inputs["node_type"]
    )
    assert features.shape[-1] == geometry_end + NUM_NODE_TYPE_FEATURES


def test_one_sample_at_a_time_matches_the_whole_block(inputs):
    """
    The evaluation script builds the vector one timestep at a time while
    training builds it for the whole simulation at once. The two must be
    the same array.
    """

    extras = {
        "geometry_features": inputs["geometry"],
        "node_type_features": inputs["node_type"],
    }

    whole = build_features(
        "time",
        inputs["X"],
        inputs["dt"],
        physics_features=inputs["physics"],
        **extras,
    )

    for timestep in range(SAMPLES):

        one = build_features(
            "time",
            inputs["X"][timestep:timestep + 1],
            inputs["dt"][timestep:timestep + 1],
            physics_features=inputs["physics"][timestep:timestep + 1],
            **extras,
        )[0]

        assert np.allclose(one, whole[timestep])


def test_a_block_with_the_wrong_number_of_nodes_is_refused(inputs):

    with pytest.raises(ValueError, match="physics features"):
        build_features(
            "time",
            inputs["X"],
            inputs["dt"],
            physics_features=inputs["physics"][:, :NODES - 1],
        )


def test_a_block_with_the_wrong_width_is_refused(inputs):

    with pytest.raises(ValueError, match="physics features"):
        build_features(
            "time",
            inputs["X"],
            inputs["dt"],
            physics_features=inputs["physics"][..., :2],
        )

    with pytest.raises(ValueError, match="geometry features"):
        build_features(
            "time",
            inputs["X"],
            inputs["dt"],
            geometry_features=inputs["geometry"][:, :2],
        )


def test_an_unknown_encoding_is_refused():

    with pytest.raises(ValueError, match="Unknown feature encoding"):
        features_size("wavelet")
