"""
Saving and loading trained models.

A checkpoint is not just weights: it also carries the feature scalings
fitted on the training block (time step, physics and geometry features)
and a description of the run, so inference never has to guess them.

The state (u, v, p) itself is not scaled by anything stored here: it is
divided by the physical scale [U_inf, U_inf, rho U_inf^2] of each
dataset (see data.normalization.normalize_simulation).
"""

from dataclasses import dataclass

import torch
from .data.normalization import (
    GeometryNormalizer,
    PhysicsNormalizer,
)


@dataclass
class CheckpointBundle:
    """
    Everything a checkpoint knows about how its model was fed.

    physics_normalizer : PhysicsNormalizer or None
        Scaling of the physics-derived features. None for models that
        were not fed them.

    geometry_normalizer : GeometryNormalizer or None
        Scaling of the static per-node geometry features. None for
        models that were not fed them.

    dt_mean, dt_std : float or None
        Scaling of the time step. None for checkpoints written before
        these were stored; callers must refuse to guess them.

    metadata : dict
        How the model was built and what it was trained on.
    """

    model: object
    physics_normalizer: PhysicsNormalizer | None
    geometry_normalizer: GeometryNormalizer | None
    dt_mean: float | None
    dt_std: float | None
    metadata: dict

def save_checkpoint(
    path,
    model,
    metadata=None,
    physics_normalizer=None,
    geometry_normalizer=None,
    dt_normalization=None,
):
    """
    Save weights, every scaling used to build the inputs and targets,
    and a description of the run.

    Parameters
    ----------
    physics_normalizer : PhysicsNormalizer or None
        Required by models fed physics-derived input features.

    geometry_normalizer : GeometryNormalizer or None
        Required by models fed the static per-node geometry features.

    dt_normalization : (mean, std) or None
        Scaling of the time-step feature, fitted on the training block.
    """
    checkpoint = {
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
    }

    if physics_normalizer is not None:
        checkpoint["physics_normalizer"] = (
            physics_normalizer.to_dict()
        )

    if geometry_normalizer is not None:
        checkpoint["geometry_normalizer"] = (
            geometry_normalizer.to_dict()
        )
    if dt_normalization is not None:

        dt_mean, dt_std = dt_normalization

        if not dt_std > 0:
            raise ValueError(
                f"dt_std must be strictly positive, got {dt_std}."
            )

        checkpoint["dt_normalization"] = {
            "mean": float(dt_mean),
            "std": float(dt_std),
        }

    torch.save(
        checkpoint,
        path,
    )


def read_checkpoint(path, model=None, device=None):
    """
    Read a checkpoint once and return everything it carries.

    Returns
    -------
    CheckpointBundle
    """

    checkpoint = torch.load(path, map_location=device, weights_only=True)

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"{path} does not contain a valid checkpoint dictionary."
        )

    if "state_dict" not in checkpoint:
        raise ValueError(
            f"{path} does not contain a model state_dict."
        )


    physics_state = checkpoint.get("physics_normalizer")

    physics_normalizer = (
        PhysicsNormalizer.from_dict(physics_state)
        if physics_state is not None
        else None
    )

    geometry_state = checkpoint.get("geometry_normalizer")

    geometry_normalizer = (
        GeometryNormalizer.from_dict(geometry_state)
        if geometry_state is not None
        else None
    )



    dt_normalization = checkpoint.get("dt_normalization") or {}

    if model is not None:
        model.load_state_dict(checkpoint["state_dict"])

    return CheckpointBundle(
        model=model,
        physics_normalizer=physics_normalizer,
        geometry_normalizer=geometry_normalizer,
        dt_mean=dt_normalization.get("mean"),
        dt_std=dt_normalization.get("std"),
        metadata=checkpoint.get("metadata", {}),
    )



def load_physics_normalizer(
    path,
    device=None,
    required=False,
):
    """
    The PhysicsNormalizer stored in a checkpoint, or None.

    With required=True, raise when the checkpoint does not carry one.
    """

    physics_normalizer = read_checkpoint(
        path,
        device=device,
    ).physics_normalizer

    if physics_normalizer is None and required:
        raise ValueError(
            f"{path} has no physics normalizer stored in it, but this "
            "model requires physics-derived input features."
        )

    return physics_normalizer


def load_geometry_normalizer(
    path,
    device=None,
    required=False,
):
    """
    The GeometryNormalizer stored in a checkpoint, or None.

    With required=True, raise when the checkpoint does not carry one.
    """

    geometry_normalizer = read_checkpoint(
        path,
        device=device,
    ).geometry_normalizer

    if geometry_normalizer is None and required:
        raise ValueError(
            f"{path} has no geometry normalizer stored in it, but this "
            "model requires geometry-derived input features."
        )

    return geometry_normalizer
