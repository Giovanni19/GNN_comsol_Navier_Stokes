"""
Hard imposition of the Dirichlet boundary conditions.

Besides the projection itself, the residual is checked the only way
that catches a projection landing on the wrong nodes: two predictions
that differ ONLY at the Dirichlet nodes must give exactly the same
physics loss.
"""

import numpy as np
import pytest
import torch

from gnn_comsol.boundary import (
    apply_velocity_dirichlet,
    boundary_consistency,
    velocity_dirichlet_violation,
)
from gnn_comsol.physics import FluidProperties, PhysicsGeometry
from gnn_comsol.training.losses import CoupledPhysicsLoss
from conftest import grid_cells, grid_mesh, wlsq_operators


# ===============================================================
# The projection itself
# ===============================================================

def test_only_the_masked_nodes_are_replaced():

    velocity = torch.arange(10, dtype=torch.float64).reshape(5, 2)

    mask = torch.tensor([True, False, False, True, False])

    bc = torch.full((5, 2), -1.0, dtype=torch.float64)

    result = apply_velocity_dirichlet(velocity, mask, bc)

    assert torch.equal(result[mask], bc[mask])
    assert torch.equal(result[~mask], velocity[~mask])


def test_an_integer_mask_is_accepted():
    """MATLAB writes the mask as 0/1, and h5py hands it back as such."""

    velocity = torch.zeros(3, 2)
    bc = torch.ones(3, 2)

    result = apply_velocity_dirichlet(
        velocity,
        torch.tensor([1, 0, 1], dtype=torch.uint8),
        bc,
    )

    assert torch.equal(result[:, 0], torch.tensor([1.0, 0.0, 1.0]))


def test_the_dirichlet_nodes_receive_no_gradient():
    """
    The replacement is against a constant, so nothing downstream can
    push the network at a node whose value is given.
    """

    velocity = torch.zeros(4, 2, requires_grad=True)

    mask = torch.tensor([True, False, True, False])

    result = apply_velocity_dirichlet(
        velocity,
        mask,
        torch.ones(4, 2),
    )

    result.sum().backward()

    assert torch.equal(
        velocity.grad[mask],
        torch.zeros(2, 2),
    )

    assert torch.all(velocity.grad[~mask] == 1.0)


@pytest.mark.parametrize(
    "velocity, mask, bc",
    [
        # velocity is not (N, 2)
        (torch.zeros(4, 3), torch.ones(4, dtype=torch.bool),
         torch.zeros(4, 3)),
        # bc covers a different number of nodes
        (torch.zeros(4, 2), torch.ones(4, dtype=torch.bool),
         torch.zeros(3, 2)),
        # the mask covers a different number of nodes
        (torch.zeros(4, 2), torch.ones(3, dtype=torch.bool),
         torch.zeros(4, 2)),
        # the mask is not one value per node
        (torch.zeros(4, 2), torch.ones(4, 2, dtype=torch.bool),
         torch.zeros(4, 2)),
    ],
)
def test_a_shape_mismatch_is_refused(velocity, mask, bc):
    """
    Broadcasting one node's boundary value over the mesh would produce
    a field that looks plausible and is not, so it must fail loudly.
    """

    with pytest.raises(ValueError):
        apply_velocity_dirichlet(velocity, mask, bc)


def test_violation_measures_only_the_masked_nodes():

    velocity = torch.tensor(
        [[3.0, 4.0], [100.0, 100.0]],
        dtype=torch.float64,
    )

    mask = torch.tensor([True, False])

    bc = torch.zeros(2, 2, dtype=torch.float64)

    violation = velocity_dirichlet_violation(velocity, mask, bc)

    # sqrt(mean(3^2, 4^2)) over the one masked node
    assert violation["rms"] == pytest.approx(np.sqrt(12.5))
    assert violation["max"] == pytest.approx(4.0)
    assert violation["num_nodes"] == 1


def test_violation_of_an_empty_mask_is_not_zero():
    """
    Zero would read as "the boundary is perfect". There is no boundary.
    """

    violation = velocity_dirichlet_violation(
        torch.zeros(3, 2),
        torch.zeros(3, dtype=torch.bool),
        torch.zeros(3, 2),
    )

    assert np.isnan(violation["rms"])
    assert violation["num_nodes"] == 0


# ===============================================================
# The projection where it actually runs: inside CoupledPhysicsLoss
# ===============================================================

ROWS, COLS, SAMPLES = 4, 5, 3


@pytest.fixture
def coupled():
    """
    One small simulation, its WLSQ geometry, and a coupled batch.

    Velocity mask: left column plus top and bottom rows (walls +
    inlet); pressure mask: the right column (outlet).
    """

    rng = np.random.default_rng(1)

    edge_index, pos = grid_mesh(ROWS, COLS, spacing=0.25)
    neighbors, G_wlsq = wlsq_operators(edge_index, pos)
    num_nodes = ROWS * COLS

    geometry = PhysicsGeometry(
        neighbors=neighbors,
        G_wlsq=G_wlsq,
        cell_index=grid_cells(ROWS, COLS),
        pos=pos,
        num_nodes=num_nodes,
        fluid=FluidProperties(rho=1.0, mu=0.01, source="test"),
        simulation_id=0,
    )

    rows, cols = np.divmod(np.arange(num_nodes), COLS)
    velocity_mask = (cols == 0) | (rows == 0) | (rows == ROWS - 1)
    pressure_mask = cols == COLS - 1

    def tensor(array):
        return torch.as_tensor(array, dtype=torch.float32)

    batch = {
        "velocity_X": tensor(rng.normal(size=(SAMPLES, num_nodes, 4))),
        "dt_physical": tensor(np.full(SAMPLES, 0.01)),
        "simulation_ids": torch.zeros(SAMPLES, dtype=torch.long),
        "velocity_bc": tensor(rng.normal(size=(SAMPLES, num_nodes, 2))),
        "velocity_dirichlet_mask": torch.as_tensor(
            np.tile(velocity_mask, (SAMPLES, 1))
        ),
        "pressure_bc": tensor(rng.normal(size=(SAMPLES, num_nodes))),
        "pressure_dirichlet_mask": torch.as_tensor(
            np.tile(pressure_mask, (SAMPLES, 1))
        ),
    }

    preds = {
        "velocity_preds": tensor(rng.normal(size=(SAMPLES, num_nodes, 2))),
        "pressure_preds": tensor(rng.normal(size=(SAMPLES, num_nodes, 1))),
    }

    return geometry, batch, preds, velocity_mask, pressure_mask


def coupled_loss(geometry, enforce):

    return CoupledPhysicsLoss(
        geometries={0: geometry},
        state_scales={0: np.array([1.0, 1.0, 1.0])},
        enforce_boundary_values=enforce,
    )


def residuals(loss, batch, preds):

    components, _ = loss(
        batch["velocity_X"],
        preds["velocity_preds"],
        preds["pressure_preds"],
        batch["dt_physical"],
        batch["simulation_ids"],
        batch["velocity_bc"],
        batch["velocity_dirichlet_mask"],
        batch["pressure_bc"],
        batch["pressure_dirichlet_mask"],
    )

    return {name: float(value) for name, value in components.items()}


def perturbed_at(preds, velocity_mask, pressure_mask):

    perturbed = {name: value.clone() for name, value in preds.items()}
    perturbed["velocity_preds"][:, velocity_mask] += 17.0
    perturbed["pressure_preds"][:, pressure_mask] += 17.0

    return perturbed


def test_the_residual_ignores_what_is_predicted_at_the_boundary(coupled):
    """
    Two predictions differing ONLY at the Dirichlet nodes must give the
    same residuals: both are replaced by the prescribed values first.
    """

    geometry, batch, preds, velocity_mask, pressure_mask = coupled

    loss = coupled_loss(geometry, enforce=True)

    reference = residuals(loss, batch, preds)
    perturbed = residuals(
        loss, batch, perturbed_at(preds, velocity_mask, pressure_mask)
    )

    for name, value in reference.items():
        assert perturbed[name] == pytest.approx(value, rel=1e-6)


def test_without_enforcement_the_boundary_does_change_the_residual(coupled):
    """The counterpart, so the test above cannot pass by accident."""

    geometry, batch, preds, velocity_mask, pressure_mask = coupled

    loss = coupled_loss(geometry, enforce=False)

    reference = residuals(loss, batch, preds)
    perturbed = residuals(
        loss, batch, perturbed_at(preds, velocity_mask, pressure_mask)
    )

    assert perturbed["momentum"] != pytest.approx(reference["momentum"])


# ===============================================================
# The dataset itself: prescribed value vs solved field
# ===============================================================

def test_a_consistent_export_disagrees_by_nothing():

    mask = np.array([True, True, False, False])

    solved = np.zeros((3, 4, 2))
    solved[:, :2, :] = 0.0
    solved[:, 2:, :] = 5.0

    prescribed = np.zeros((3, 4, 2))

    consistency = boundary_consistency(prescribed, mask, solved)

    assert consistency["largest"] == 0.0
    assert consistency["num_nodes"] == 2
    assert consistency["scale"] == 5.0


def test_a_corner_written_with_the_inlet_value_is_caught():
    """
    The real failure this check exists for: node 0 is a wall AND the
    inlet, first_database.m writes the inlet last, so the prescribed
    value there is the inflow while COMSOL solved it as no-slip.
    """

    mask = np.array([True, True, False])

    solved = np.zeros((2, 3, 2))
    solved[:, 2, 0] = 2.0          # interior, moving

    prescribed = np.zeros((2, 3, 2))
    prescribed[:, 0, 0] = 1.0      # the corner, given the inlet value

    consistency = boundary_consistency(prescribed, mask, solved)

    assert consistency["largest"] == pytest.approx(1.0)
    assert consistency["relative"] == pytest.approx(0.5)


def test_mismatched_shapes_are_refused():

    with pytest.raises(ValueError, match="same"):
        boundary_consistency(
            np.zeros((2, 3, 2)),
            np.array([True, True, False]),
            np.zeros((3, 3, 2)),
        )
