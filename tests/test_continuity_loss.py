"""
Check the continuity residual on a real COMSOL snapshot. The residual
of the ground truth is the discretization floor.

    python scripts/test_continuity_loss.py
"""

import time

import torch

from gnn_comsol.data.loading import load_data
from gnn_comsol.physics import (
    PhysicsGeometry,
    continuity_loss,
    continuity_residual,
)

dataset_path = (
    "C:/Users/giovanni/.comsol/v64/llmatlab/"
    "channel2d_manual_stabilization_200R.mat"
)


# LOAD DATASET

data = load_data(
    dataset_path,
    skip_initial=0,
)


# BUILD THE MESH OPERATORS

start = time.perf_counter()

geometry = PhysicsGeometry.from_simulation(data)

if geometry is None:
    raise RuntimeError(
        "Dataset does not contain WLSQ geometry. "
        "Regenerate it with first_database.m."
    )

operators = geometry.operators(dtype=torch.float64)
cells = geometry.cells(dtype=torch.float64)

build_seconds = time.perf_counter() - start

print("\n========================================")
print("MESH")
print("========================================")

print("nodes          :", data.num_nodes)
print("cells          :", geometry.num_cells)
print("non-zeros in Dx:", operators.dx._nnz())
print(f"build time     : {build_seconds:.3f} s (once per mesh)")


# MATLAB timestep 100 -> Python index 99. u and v require gradients,
# like network outputs.

test_step = 99

u = torch.tensor(
    data.X_input[test_step, :, 0],
    dtype=torch.float64,
    requires_grad=True,
)

v = torch.tensor(
    data.X_input[test_step, :, 1],
    dtype=torch.float64,
    requires_grad=True,
)


# CONTINUITY RESIDUAL

residual = continuity_residual(
    u,
    v,
    operators,
    cells,
)


print("\n========================================")
print("CONTINUITY RESIDUAL TEST")
print("========================================")

print("Residual shape:", residual.shape)

print("\nResidual statistics:")
print("min      =", residual.min().item())
print("max      =", residual.max().item())
print("mean     =", residual.mean().item())
print("abs mean =", residual.abs().mean().item())
print(
    "RMS      =",
    torch.sqrt(torch.mean(residual.square())).item(),
)


# Scale of the residual, against the two terms that cancel in it

from gnn_comsol.physics import (  # noqa: E402
    node_gradient_to_cell,
    wlsq_gradient,
)

with torch.no_grad():

    gradient = node_gradient_to_cell(
        wlsq_gradient(
            torch.stack((u.detach(), v.detach()), dim=1),
            operators,
        ),
        cells.cell_index,
    )

    du_dx = gradient[:, 0, 0]
    dv_dy = gradient[:, 1, 1]

    scale = torch.maximum(du_dx.abs(), dv_dy.abs())

    print("\nRelative to the terms that cancel:")
    print(
        "mean |du/dx| =",
        du_dx.abs().mean().item(),
    )
    print(
        "mean |dv/dy| =",
        dv_dy.abs().mean().item(),
    )
    print(
        "mean |div(u)| / mean max(|du/dx|, |dv/dy|) =",
        (residual.abs().mean() / scale.mean()).item(),
    )


# CONTINUITY LOSS

loss = continuity_loss(
    u,
    v,
    operators,
    cells,
)

print("\nContinuity loss:")
print(loss.item())


# BACKPROPAGATION TEST

loss.backward()


print("\n========================================")
print("BACKPROPAGATION TEST")
print("========================================")

print("u.grad exists:", u.grad is not None)
print("v.grad exists:", v.grad is not None)

print("u.grad finite:", torch.isfinite(u.grad).all().item())
print("v.grad finite:", torch.isfinite(v.grad).all().item())

print(
    "u.grad abs mean:",
    u.grad.abs().mean().item(),
)

print(
    "v.grad abs mean:",
    v.grad.abs().mean().item(),
)

print(
    "u.grad max abs:",
    u.grad.abs().max().item(),
)

print(
    "v.grad max abs:",
    v.grad.abs().max().item(),
)
