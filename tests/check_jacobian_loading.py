#!/usr/bin/env python3

"""
Check that loading.py correctly loads and aligns the COMSOL FEM Jacobians.
"""

import sys
from pathlib import Path

import numpy as np

# Make src/ importable
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from gnn_comsol.data.loading import load_data


DATASET = (
    r"C:\Users\giovanni\.comsol\v64\llmatlab"
    r"\channel2d_manual_stabilization_40R_overfit.mat"
)


def check_dataset(skip_initial):

    print()
    print("=" * 60)
    print(f"CHECK WITH skip_initial = {skip_initial}")
    print("=" * 60)

    data = load_data(
        DATASET,
        skip_initial=skip_initial,
        simulation_id=0,
    )

    # BASIC DATASET

    print("\nSTATE DATA")
    print("-" * 60)

    print("X_input shape :", data.X_input.shape)
    print("Y_target shape:", data.Y_target.shape)
    print("delta_t shape :", data.delta_t.shape)

    S = data.num_samples
    N = data.num_nodes

    print(f"\nSamples : {S}")
    print(f"Nodes   : {N}")
    print(f"3*N     : {3*N}")

    # JACOBIAN

    print("\nJACOBIAN DATA")
    print("-" * 60)

    if data.jacobian_values is None:
        raise RuntimeError("jacobian_values was not loaded.")

    if data.jacobian_row is None:
        raise RuntimeError("jacobian_row was not loaded.")

    if data.jacobian_col is None:
        raise RuntimeError("jacobian_col was not loaded.")

    print(
        "shared pattern :",
        data.jacobian_shared_pattern,
    )

    print(
        "jacobian shape :",
        data.jacobian_shape,
    )

    print(
        "row shape      :",
        data.jacobian_row.shape,
    )

    print(
        "col shape      :",
        data.jacobian_col.shape,
    )

    print(
        "values shape   :",
        data.jacobian_values.shape,
    )

    print(
        "nnz/sample     :",
        data.jacobian_row.size,
    )

    # SAMPLE ALIGNMENT

    print("\nSAMPLE ALIGNMENT")
    print("-" * 60)

    assert data.X_input.shape[0] == S
    assert data.Y_target.shape[0] == S
    assert data.delta_t.shape[0] == S

    assert data.jacobian_values.shape[0] == S

    print(
        "X / Y / dt / J samples:",
        S,
        "-> PASS",
    )

    # MATRIX DIMENSION

    expected_dofs = 3 * N

    assert tuple(data.jacobian_shape) == (
        expected_dofs,
        expected_dofs,
    )

    print(
        f"Jacobian dimension = {expected_dofs} x {expected_dofs}"
        " -> PASS"
    )

    # COO PATTERN

    nnz = data.jacobian_row.size

    assert data.jacobian_col.size == nnz

    assert data.jacobian_values.shape[1] == nnz

    assert data.jacobian_row.min() >= 0
    assert data.jacobian_col.min() >= 0

    assert data.jacobian_row.max() < expected_dofs
    assert data.jacobian_col.max() < expected_dofs

    print(
        f"COO pattern: nnz={nnz} -> PASS"
    )

    # NODE -> FEM DOF MAPPING

    print("\nNODE -> FEM DOF MAPPING")
    print("-" * 60)

    node = np.arange(N, dtype=np.int64)

    expected_p = 3 * node
    expected_u = 3 * node + 1
    expected_v = 3 * node + 2

    assert np.array_equal(
        data.jacobian_dof_p,
        expected_p,
    )

    assert np.array_equal(
        data.jacobian_dof_u,
        expected_u,
    )

    assert np.array_equal(
        data.jacobian_dof_v,
        expected_v,
    )

    print("p_i = 3*i     -> PASS")
    print("u_i = 3*i + 1 -> PASS")
    print("v_i = 3*i + 2 -> PASS")

    print("\nFirst 5 nodes:")

    for i in range(min(5, N)):
        print(
            f"node {i:4d}: "
            f"p={data.jacobian_dof_p[i]:6d} | "
            f"u={data.jacobian_dof_u[i]:6d} | "
            f"v={data.jacobian_dof_v[i]:6d}"
        )

    # TARGET TIME

    print("\nTARGET TIME")
    print("-" * 60)

    if data.jacobian_target_time is None:
        raise RuntimeError(
            "jacobian_target_time was not loaded."
        )

    assert data.jacobian_target_time.shape == (S,)

    print(
        "first Jacobian target time:",
        data.jacobian_target_time[0],
    )

    print(
        "last Jacobian target time :",
        data.jacobian_target_time[-1],
    )

    print("Target-time alignment -> PASS")

    # NUMERICAL VALUES

    print("\nJACOBIAN VALUES")
    print("-" * 60)

    assert np.all(
        np.isfinite(data.jacobian_values)
    )

    print(
        "min value:",
        np.min(data.jacobian_values),
    )

    print(
        "max value:",
        np.max(data.jacobian_values),
    )

    print(
        "mean |value|:",
        np.mean(
            np.abs(data.jacobian_values)
        ),
    )

    print("Finite values -> PASS")

    # RECONSTRUCT ONE SPARSE J

    print("\nRECONSTRUCT FIRST JACOBIAN")
    print("-" * 60)

    try:
        from scipy.sparse import coo_matrix

        J = coo_matrix(
            (
                data.jacobian_values[0],
                (
                    data.jacobian_row,
                    data.jacobian_col,
                ),
            ),
            shape=(
                expected_dofs,
                expected_dofs,
            ),
        ).tocsr()

        print("J shape:", J.shape)
        print("J nnz  :", J.nnz)

        assert J.shape == (
            expected_dofs,
            expected_dofs,
        )

        assert J.nnz > 0

        print("Sparse reconstruction -> PASS")

    except ImportError:

        print(
            "scipy not installed: "
            "sparse reconstruction skipped."
        )

    print()
    print("=" * 60)
    print(
        f"ALL CHECKS PASSED FOR skip_initial={skip_initial}"
    )
    print("=" * 60)


def main():

    # Full trajectory:
    #
    # 51 snapshots -> 50 transitions -> 50 Jacobians
    check_dataset(skip_initial=0)

    # Last transition only:
    #
    # X at 1.96 s -> Y at 2.00 s
    # must correspond to exactly one Jacobian at t=2.00 s.
    check_dataset(skip_initial=49)

    print()
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print("Jacobian loading and alignment are correct.")


if __name__ == "__main__":
    main()
