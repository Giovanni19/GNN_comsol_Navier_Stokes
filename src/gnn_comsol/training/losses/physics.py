"""Coupled physics loss used during training."""

from dataclasses import dataclass

import torch

from gnn_comsol.data.normalization import (
    PRESSURE_COLUMNS,
    STATE_COLUMNS,
    VELOCITY_COLUMNS,
)
from gnn_comsol.physics import (
    continuity_loss,
    momentum_loss,
)



@dataclass
class CoupledPhysicsLoss:

    geometries: dict
    state_scales: dict

    predict_velocity_delta: bool = False
    predict_pressure_delta: bool = False

    continuity_reference: float = 1.0
    momentum_reference: float = 1.0

    enforce_boundary_values: bool = True

    def __post_init__(self):

        if not self.geometries:
            raise ValueError(
                "The coupled PDE residuals need the WLSQ geometry "
                "of every simulation."
            )

        if self.continuity_reference <= 0.0:
            raise ValueError(
                "continuity_reference must be positive."
            )

        if self.momentum_reference <= 0.0:
            raise ValueError(
                "momentum_reference must be positive."
            )

    @property
    def terms(self):
        return ["continuity", "momentum"]    

    def velocity_to_physical(
        self,
        normalized,
        current,
        state_scale,
    ):

        velocity_scale = state_scale[VELOCITY_COLUMNS]

        if not self.predict_velocity_delta:
            return normalized * velocity_scale

        delta_physical = normalized * velocity_scale

        return current + delta_physical

    def pressure_to_physical(
        self,
        normalized,
        current,
        state_scale,
    ):

        pressure_scale = state_scale[PRESSURE_COLUMNS]

        if not self.predict_pressure_delta:
            return normalized * pressure_scale

        delta_physical = normalized * pressure_scale

        return current + delta_physical
    def __call__(
        self,
        velocity_X,
        velocity_preds,
        pressure_preds,
        dt_physical,
        simulation_ids,
        velocity_bc,
        velocity_dirichlet_mask,
        pressure_bc,
        pressure_dirichlet_mask,
    ):
        """
        Coupled physics loss for one synchronized batch.

        velocity_preds : (B, N, 2)
            Normalized velocity-network predictions.

        pressure_preds : (B, N, 1)
            Normalized pressure-network predictions.

        All samples in the batch belong to the same simulation.
        """

        simulation_id = int(simulation_ids[0].item())
        state_scale_np = self.state_scales.get(simulation_id)

        if state_scale_np is None:
            raise KeyError(
                f"Simulation {simulation_id} has no physical state scale."
            )

        state_scale = torch.as_tensor(
            state_scale_np,
            device=velocity_preds.device,
            dtype=velocity_preds.dtype,
        )

        if not torch.all(simulation_ids == simulation_ids[0]):
            raise ValueError(
                "A coupled batch must contain samples from only "
                "one simulation."
            )

        geometry = self.geometries.get(simulation_id)

        if geometry is None:
            raise KeyError(
                f"Simulation {simulation_id} has no physics geometry."
            )

        device = velocity_preds.device
        dtype = velocity_preds.dtype

        operators = geometry.operators(
            device=device,
            dtype=dtype,
        )

        cells = geometry.cells(
            device=device,
            dtype=dtype,
        )

        fluid = geometry.require_fluid()

        continuity_values = []
        momentum_values = []

        # Physics residuals are evaluated transition by transition

        for i in range(velocity_preds.shape[0]):

            # Current physical state: features start with [u, v, p].
            state_now = (
                velocity_X[i, :, STATE_COLUMNS]
                * state_scale
            )

            velocity_now = state_now[:, VELOCITY_COLUMNS]
            pressure_now = state_now[:, PRESSURE_COLUMNS]

            # Predicted next velocity -> physical units

            velocity_next = self.velocity_to_physical(
                velocity_preds[i],
                velocity_now,
                state_scale,
            )

            # Predicted next pressure -> physical units

            pressure_next = self.pressure_to_physical(
                pressure_preds[i],
                pressure_now,
                state_scale,
            ).squeeze(-1)
            # Continuity

            continuity_values.append(
                continuity_loss(
                    velocity_next[:, 0],
                    velocity_next[:, 1],
                    operators,
                    cells,
                )
            )

            momentum_values.append(
                momentum_loss(
                    velocity_now,
                    velocity_next,
                    pressure_next,
                    dt_physical[i],
                    fluid,
                    operators,
                    cells,
                )
            )

        components = {}

        if continuity_values:
            components["continuity"] = torch.stack(
                continuity_values
            ).mean()

        if momentum_values:
            components["momentum"] = torch.stack(
                momentum_values
            ).mean()

        # Weighted physics loss

        normalized_components = {
            "continuity": (
                components["continuity"]
                / self.continuity_reference
            ),
            "momentum": (
                components["momentum"]
                / self.momentum_reference
            ),
        }

        return components, normalized_components