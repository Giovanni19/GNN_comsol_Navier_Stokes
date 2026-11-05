"""Checkpoints must carry the scalings their model was trained with."""

import numpy as np
import pytest
import torch

from gnn_comsol.checkpoints import read_checkpoint, save_checkpoint
from gnn_comsol.data.normalization import (
    NUM_PHYSICS_FEATURES,
    PhysicsNormalizer,
)


def test_raw_state_dict_is_refused(tmp_path):
    """A bare state_dict carries none of the scalings."""

    path = tmp_path / "legacy.pth"

    torch.save({"weight": torch.zeros(2)}, path)

    with pytest.raises(ValueError, match="state_dict"):
        read_checkpoint(path)


def test_scalings_survive_the_round_trip(tmp_path):

    path = tmp_path / "model.pth"

    model = torch.nn.Linear(3, 2)

    physics_normalizer = PhysicsNormalizer(
        np.arange(NUM_PHYSICS_FEATURES, dtype=float),
        np.ones(NUM_PHYSICS_FEATURES),
    )

    save_checkpoint(
        path,
        model,
        metadata={"network": "velocity"},
        physics_normalizer=physics_normalizer,
        dt_normalization=(0.5, 0.1),
    )

    bundle = read_checkpoint(path)

    assert bundle.metadata == {"network": "velocity"}
    assert bundle.geometry_normalizer is None
    assert bundle.dt_mean == pytest.approx(0.5)
    assert bundle.dt_std == pytest.approx(0.1)
    assert np.allclose(
        bundle.physics_normalizer.mean, physics_normalizer.mean
    )
