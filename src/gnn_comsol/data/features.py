"""
Node-feature construction.

Base encodings:
    time         -> [u, v, p, dt]
    time_fourier -> [u, v, p, dt, Fourier(dt)]

Optional blocks are appended in this order:
    physics | geometry | node type
"""

import numpy as np

from .normalization import (
    NUM_GEOMETRY_FEATURES,
    NUM_PHYSICS_FEATURES,
)


FEATURE_SIZES = {
    "time": 4,
    "time_fourier": 12,
}

NODE_TYPE_FEATURE_NAMES = (
    "node_interior",
    "node_wall",
    "node_inlet",
    "node_outlet",
)

NUM_NODE_TYPE_FEATURES = len(NODE_TYPE_FEATURE_NAMES)


# Sizes

def features_size(
    encoding,
    use_physics_features=False,
    use_geometry_features=False,
    use_node_type_features=False,
):
    if encoding not in FEATURE_SIZES:
        raise ValueError(
            f"Unknown feature encoding {encoding!r}. "
            f"Expected one of {sorted(FEATURE_SIZES)}."
        )

    return (
        FEATURE_SIZES[encoding]
        + NUM_PHYSICS_FEATURES * use_physics_features
        + NUM_GEOMETRY_FEATURES * use_geometry_features
        + NUM_NODE_TYPE_FEATURES * use_node_type_features
    )


# Time encoding

def add_time_feature(X, delta_t):
    """(S, N, 3) -> (S, N, 4)."""

    dt = np.broadcast_to(
        delta_t[:, None, None],
        (*X.shape[:2], 1),
    )

    return np.concatenate([X, dt], axis=-1)


def add_time_fourier_features(
    X,
    delta_t,
    num_frequencies=4,
):
    """Append dt and sin/cos encodings of normalized dt."""

    num_samples, num_nodes = X.shape[:2]
    frequencies = 2.0 ** np.arange(num_frequencies)

    blocks = [
        X,
        np.broadcast_to(
            delta_t[:, None, None],
            (num_samples, num_nodes, 1),
        ),
    ]

    for frequency in frequencies:
        angle = 2.0 * np.pi * frequency * delta_t

        blocks.extend([
            np.broadcast_to(
                np.sin(angle)[:, None, None],
                (num_samples, num_nodes, 1),
            ),
            np.broadcast_to(
                np.cos(angle)[:, None, None],
                (num_samples, num_nodes, 1),
            ),
        ])

    return np.concatenate(blocks, axis=-1)


def _base_features(
    encoding,
    X,
    delta_t,
    num_frequencies,
):
    if encoding == "time":
        return add_time_feature(X, delta_t)

    if encoding == "time_fourier":
        return add_time_fourier_features(
            X,
            delta_t,
            num_frequencies,
        )

    raise ValueError(
        f"Unknown feature encoding {encoding!r}. "
        f"Expected one of {sorted(FEATURE_SIZES)}."
    )


# Feature helpers

def _dynamic_block(
    values,
    name,
    num_samples,
    num_nodes,
    width,
):
    """Validate a dynamic (S, N, F) feature block."""

    if values is None:
        return None

    values = np.asarray(values)

    expected = (num_samples, num_nodes, width)

    if values.shape != expected:
        raise ValueError(
            f"{name} must have shape {expected}, "
            f"got {values.shape}."
        )

    return values


def _static_block(
    values,
    name,
    num_samples,
    num_nodes,
    width,
):
    """Validate (N, F) features and broadcast them over samples."""

    if values is None:
        return None

    values = np.asarray(values)

    expected = (num_nodes, width)

    if values.shape != expected:
        raise ValueError(
            f"{name} must have shape {expected}, "
            f"got {values.shape}."
        )

    return np.broadcast_to(
        values[None, :, :],
        (num_samples, num_nodes, width),
    )


def _append(features, *blocks):
    """Append every non-None feature block."""

    return np.concatenate(
        [features, *(block for block in blocks if block is not None)],
        axis=-1,
    )


# Node types

def build_node_type_features(
    wall_mask,
    inlet_mask,
    outlet_mask,
):
    """
    Mutually exclusive one-hot node types.

    Priority:
        wall > inlet > outlet > interior
    """

    wall = np.asarray(wall_mask, dtype=bool).reshape(-1)
    inlet = np.asarray(inlet_mask, dtype=bool).reshape(-1)
    outlet = np.asarray(outlet_mask, dtype=bool).reshape(-1)

    if not (wall.shape == inlet.shape == outlet.shape):
        raise ValueError(
            "wall, inlet and outlet masks must have the same shape."
        )

    inlet &= ~wall
    outlet &= ~wall & ~inlet
    interior = ~(wall | inlet | outlet)

    features = np.stack(
        [interior, wall, inlet, outlet],
        axis=-1,
    ).astype(np.float32)

    if not np.all(features.sum(axis=1) == 1):
        raise ValueError(
            "Every node must have exactly one node type."
        )

    return features


# Velocity/general features

def build_features(
    kind,
    X_norm,
    dt_norm,
    physics_features=None,
    geometry_features=None,
    node_type_features=None,
    num_frequencies=4,
):
    """
    Build:
        [base | physics | geometry | node type]
    """

    features = _base_features(
        kind,
        X_norm,
        dt_norm,
        num_frequencies,
    )

    num_samples, num_nodes = features.shape[:2]

    physics = _dynamic_block(
        physics_features,
        "physics features",
        num_samples,
        num_nodes,
        NUM_PHYSICS_FEATURES,
    )

    geometry = _static_block(
        geometry_features,
        "geometry features",
        num_samples,
        num_nodes,
        NUM_GEOMETRY_FEATURES,
    )

    node_types = _static_block(
        node_type_features,
        "node type features",
        num_samples,
        num_nodes,
        NUM_NODE_TYPE_FEATURES,
    )

    if node_type_features is not None:
        if not np.all(
            np.asarray(node_type_features).sum(axis=1) == 1
        ):
            raise ValueError(
                "Every node must have exactly one node type."
            )

    features = _append(
        features,
        physics,
        geometry,
        node_types,
    )

    expected = features_size(
        kind,
        physics_features is not None,
        geometry_features is not None,
        node_type_features is not None,
    )

    if features.shape[-1] != expected:
        raise ValueError(
            f"Assembled {features.shape[-1]} features, "
            f"expected {expected}."
        )

    return features
