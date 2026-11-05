"""
Load COMSOL datasets exported as MATLAB v7.3/HDF5 files.

State convention:
    X[..., 0] = u
    X[..., 1] = v
    X[..., 2] = p

Each returned sample represents:
    X_input[i] -> Y_target[i]

Optional data include physics/geometry features, boundary conditions,
WLSQ operators, fluid properties, and COMSOL FEM Jacobians.
"""

from dataclasses import dataclass

import h5py
import numpy as np

from .normalization import NUM_GEOMETRY_FEATURES, NUM_PHYSICS_FEATURES


@dataclass
class RawDataset:
    X_input: np.ndarray
    Y_target: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray
    delta_t: np.ndarray
    pos: np.ndarray

    wall_mask: np.ndarray | None = None
    inlet_mask: np.ndarray | None = None
    outlet_mask: np.ndarray | None = None

    neighbors: list[np.ndarray] | None = None
    G_wlsq: list[np.ndarray] | None = None
    cell_index: np.ndarray | None = None

    rho: float | None = None
    mu: float | None = None
    U_inf: float | None = None
    R: float | None = None

    physics_features: np.ndarray | None = None
    geometry_features: np.ndarray | None = None

    velocity_dirichlet_mask: np.ndarray | None = None
    velocity_bc_target: np.ndarray | None = None

    pressure_dirichlet_mask: np.ndarray | None = None
    pressure_bc_target: np.ndarray | None = None

    simulation_id: int | None = None
    file_path: str | None = None

    jacobian_shared_pattern: bool | None = None
    jacobian_row: np.ndarray | None = None
    jacobian_col: np.ndarray | None = None
    jacobian_values: np.ndarray | None = None
    jacobian_shape: np.ndarray | None = None

    jacobian_dof_p: np.ndarray | None = None
    jacobian_dof_u: np.ndarray | None = None
    jacobian_dof_v: np.ndarray | None = None

    jacobian_target_time: np.ndarray | None = None

    @property
    def num_samples(self):
        return len(self.X_input)

    @property
    def num_nodes(self):
        return self.X_input.shape[1]

    @property
    def num_edges(self):
        return self.edge_index.shape[1]


# HDF5 helpers

def _load_optional_array(f, key):
    return np.array(f[key]) if key in f else None


def _load_optional_scalar(f, key):
    if key not in f:
        return None

    value = np.asarray(f[key]).reshape(-1)

    if value.size != 1:
        raise ValueError(
            f"{key} must contain one value, got {value.size}."
        )

    return float(value[0])


def _load_matlab_cell_array(f, key):
    if key not in f:
        return None

    refs = np.asarray(f[key]).reshape(-1)
    return [np.asarray(f[ref]) for ref in refs]


def _int_vector(value):
    if value is None:
        return None

    return np.asarray(value).reshape(-1).astype(np.int64)


def _bool_vector(value):
    if value is None:
        return None

    return np.asarray(value).reshape(-1).astype(bool)


# WLSQ

def _process_wlsq(
    neighbors,
    G_wlsq,
    cell_index,
    num_nodes,
    file_path,
):
    """Convert MATLAB/HDF5 WLSQ data to the Python layout."""

    if neighbors is not None:

        if len(neighbors) != num_nodes:
            raise ValueError(
                f"{file_path}: neighbors_python has {len(neighbors)} "
                f"entries, expected {num_nodes}."
            )

        neighbors = [
            np.asarray(neigh).reshape(-1).astype(np.int64)
            for neigh in neighbors
        ]

        for node_i, neigh in enumerate(neighbors):

            if neigh.size < 2:
                raise ValueError(
                    f"{file_path}: node {node_i} has only "
                    f"{neigh.size} WLSQ neighbor(s)."
                )

            if np.any(neigh < 0) or np.any(neigh >= num_nodes):
                raise ValueError(
                    f"{file_path}: invalid WLSQ neighbors "
                    f"at node {node_i}."
                )

    if G_wlsq is not None:

        if neighbors is None:
            raise ValueError(
                f"{file_path}: G_wlsq exists but "
                "neighbors_python is missing."
            )

        if len(G_wlsq) != num_nodes:
            raise ValueError(
                f"{file_path}: G_wlsq has {len(G_wlsq)} entries, "
                f"expected {num_nodes}."
            )

        processed = []

        for node_i, G_i in enumerate(G_wlsq):

            G_i = np.asarray(G_i, dtype=np.float64)
            k_i = neighbors[node_i].size

            # MATLAB (2, k_i) -> h5py (k_i, 2)
            if G_i.shape != (k_i, 2):
                raise ValueError(
                    f"{file_path}: G_wlsq[{node_i}] has shape "
                    f"{G_i.shape}, expected ({k_i}, 2)."
                )

            processed.append(G_i.T)

        G_wlsq = processed

    if cell_index is not None:

        # MATLAB (Nc, 3) -> h5py (3, Nc)
        if cell_index.ndim != 2 or cell_index.shape[0] != 3:
            raise ValueError(
                f"{file_path}: invalid cell_index shape "
                f"{cell_index.shape}."
            )

        cell_index = cell_index.T.astype(np.int64)

        if np.any(cell_index < 0) or np.any(cell_index >= num_nodes):
            raise ValueError(
                f"{file_path}: cell_index contains invalid node indices."
            )

    return neighbors, G_wlsq, cell_index


# Validation helpers

def _validate_boundary_pair(
    values,
    mask,
    expected_shape,
    num_nodes,
    name,
    file_path,
):
    """Validate one prescribed boundary-value array and its mask."""

    if values is not None:

        if values.shape != expected_shape:
            raise ValueError(
                f"{file_path}: {name}_bc_target has shape "
                f"{values.shape}, expected {expected_shape}."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{file_path}: {name}_bc contains NaN or Inf."
            )

    if mask is not None and mask.shape != (num_nodes,):
        raise ValueError(
            f"{file_path}: {name}_dirichlet_mask has shape "
            f"{mask.shape}, expected ({num_nodes},)."
        )

    if (values is None) != (mask is None):
        raise ValueError(
            f"{file_path}: {name}_bc and "
            f"{name}_dirichlet_mask must either both exist "
            "or both be absent."
        )


def _validate_jacobian(
    *,
    file_path,
    num_nodes,
    num_samples,
    t,
    skip_initial,
    shared_pattern,
    row,
    col,
    values,
    shape,
    dof_p,
    dof_u,
    dof_v,
    target_time,
):
    """Validate the exported COMSOL FEM Jacobians."""

    if values is None:
        return

    if shared_pattern is not True:
        raise ValueError(
            f"{file_path}: FEM Jacobians must share one sparsity pattern."
        )

    if row is None or col is None:
        raise ValueError(
            f"{file_path}: Jacobian COO row/column indices are missing."
        )

    if shape is None or shape.size != 2:
        raise ValueError(
            f"{file_path}: invalid or missing jacobian_shape."
        )

    expected_dofs = 3 * num_nodes
    expected_shape = (expected_dofs, expected_dofs)

    if tuple(shape.tolist()) != expected_shape:
        raise ValueError(
            f"{file_path}: Jacobian shape "
            f"{tuple(shape.tolist())}, expected {expected_shape}."
        )

    nnz = row.size

    if col.size != nnz:
        raise ValueError(
            f"{file_path}: jacobian_row and jacobian_col "
            "have different lengths."
        )

    if values.shape != (num_samples, nnz):
        raise ValueError(
            f"{file_path}: jacobian_values has shape {values.shape}, "
            f"expected ({num_samples}, {nnz})."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{file_path}: jacobian_values contains NaN or Inf."
        )

    if (
        row.min() < 0
        or col.min() < 0
        or row.max() >= expected_dofs
        or col.max() >= expected_dofs
    ):
        raise ValueError(
            f"{file_path}: Jacobian COO indices are out of range."
        )

    # After MATLAB permutation the exported ordering must be:
    #
    # [p0,u0,v0, p1,u1,v1, ...]
    node_index = np.arange(num_nodes, dtype=np.int64)

    expected_dofs_by_field = {
        "p": 3 * node_index,
        "u": 3 * node_index + 1,
        "v": 3 * node_index + 2,
    }

    actual_dofs = {
        "p": dof_p,
        "u": dof_u,
        "v": dof_v,
    }

    for name, expected in expected_dofs_by_field.items():

        actual = actual_dofs[name]

        if actual is None or actual.shape != (num_nodes,):
            raise ValueError(
                f"{file_path}: jacobian_dof_{name} must have "
                f"shape ({num_nodes},)."
            )

        if not np.array_equal(actual, expected):
            raise ValueError(
                f"{file_path}: unexpected {name} Jacobian DOF ordering."
            )

    if target_time is None:
        raise ValueError(
            f"{file_path}: jacobian_target_time is missing."
        )

    expected_target_time = t[skip_initial + 1:]

    if target_time.shape != expected_target_time.shape:
        raise ValueError(
            f"{file_path}: jacobian_target_time has shape "
            f"{target_time.shape}, expected "
            f"{expected_target_time.shape}."
        )

    if not np.allclose(
        target_time,
        expected_target_time,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            f"{file_path}: FEM Jacobians are not aligned "
            "with Y_target in time."
        )


# Main loader

def load_data(
    file_path,
    skip_initial=0,
    simulation_id=None,
    load_jacobian=True,
):
    """Load one simulation and build one-step input/target pairs."""

    with h5py.File(file_path, "r") as f:

        X = np.asarray(f["X"])
        edge_index = np.asarray(f["edge_index"])
        edge_weight = np.asarray(f["edge_weight"])
        t = np.asarray(f["t"])
        P = np.asarray(f["P"])

        neighbors = _load_matlab_cell_array(f, "neighbors_python")
        G_wlsq = _load_matlab_cell_array(f, "G_wlsq")

        wall_mask = _load_optional_array(f, "wall_mask")
        inlet_mask = _load_optional_array(f, "inlet_mask")
        outlet_mask = _load_optional_array(f, "outlet_mask")

        rho = _load_optional_scalar(f, "rho")
        mu = _load_optional_scalar(f, "mu")
        U_inf = _load_optional_scalar(f, "U_inf")
        R = _load_optional_scalar(f, "R")

        physics_features = _load_optional_array(
            f, "physics_features"
        )
        geometry_features = _load_optional_array(
            f, "geometry_features"
        )

        velocity_bc = _load_optional_array(f, "velocity_bc")
        velocity_dirichlet_mask = _load_optional_array(
            f, "velocity_dirichlet_mask"
        )

        pressure_bc = _load_optional_array(f, "pressure_bc")
        pressure_dirichlet_mask = _load_optional_array(
            f, "pressure_dirichlet_mask"
        )

        cell_index = _load_optional_array(f, "cell_index")

        if load_jacobian:

            jacobian_shared_pattern_raw = _load_optional_array(
                f, "jacobian_shared_pattern"
            )
            jacobian_row = _load_optional_array(f, "jacobian_row")
            jacobian_col = _load_optional_array(f, "jacobian_col")
            jacobian_values = _load_optional_array(f, "jacobian_values")
            jacobian_shape = _load_optional_array(f, "jacobian_shape")

            jacobian_dof_p = _load_optional_array(f, "jacobian_dof_p")
            jacobian_dof_u = _load_optional_array(f, "jacobian_dof_u")
            jacobian_dof_v = _load_optional_array(f, "jacobian_dof_v")

            jacobian_target_time = _load_optional_array(
                f, "jacobian_target_time"
            )

        else:

            jacobian_shared_pattern_raw = None
            jacobian_row = None
            jacobian_col = None
            jacobian_values = None
            jacobian_shape = None

            jacobian_dof_p = None
            jacobian_dof_u = None
            jacobian_dof_v = None

            jacobian_target_time = None

    # MATLAB/HDF5 -> Python orientation

    # MATLAB (T, N, 3) -> h5py (3, N, T)
    X = X.transpose(2, 1, 0)

    if velocity_bc is not None:
        velocity_bc = velocity_bc.transpose(2, 1, 0)

    if pressure_bc is not None:
        pressure_bc = pressure_bc.T

    if physics_features is not None:
        physics_features = physics_features.transpose(2, 1, 0)

    edge_index = edge_index.astype(np.int64)
    edge_weight = edge_weight.squeeze()
    t = t.squeeze()

    velocity_dirichlet_mask = _bool_vector(
        velocity_dirichlet_mask
    )
    pressure_dirichlet_mask = _bool_vector(
        pressure_dirichlet_mask
    )

    # Build transitions

    num_snapshots = X.shape[0]

    if skip_initial < 0:
        raise ValueError(
            f"skip_initial must be >= 0, got {skip_initial}."
        )

    if skip_initial >= num_snapshots - 1:
        raise ValueError(
            f"skip_initial={skip_initial} leaves no transitions; "
            f"the simulation has {num_snapshots} snapshots."
        )

    X_input = X[skip_initial:-1]
    Y_target = X[skip_initial + 1:]
    delta_t = np.diff(t)[skip_initial:]

    num_samples = X_input.shape[0]
    num_nodes = X_input.shape[1]

    velocity_bc_target = (
        None
        if velocity_bc is None
        else velocity_bc[skip_initial + 1:]
    )

    pressure_bc_target = (
        None
        if pressure_bc is None
        else pressure_bc[skip_initial + 1:]
    )

    physics_input = (
        None
        if physics_features is None
        else physics_features[skip_initial:-1]
    )

    # Physics features

    if physics_input is not None:

        expected_shape = (
            num_samples,
            num_nodes,
            NUM_PHYSICS_FEATURES,
        )

        if physics_input.shape != expected_shape:
            raise ValueError(
                f"{file_path}: physics_features has shape "
                f"{physics_input.shape}, expected {expected_shape}."
            )

        if not np.all(np.isfinite(physics_input)):
            raise ValueError(
                f"{file_path}: physics_features contains NaN or Inf."
            )

    # Boundary masks

    boundary_masks = {
        "wall_mask": wall_mask,
        "inlet_mask": inlet_mask,
        "outlet_mask": outlet_mask,
    }

    for name, mask in boundary_masks.items():

        if mask is None:
            continue

        mask = _bool_vector(mask)

        if mask.shape != (num_nodes,):
            raise ValueError(
                f"{file_path}: {name} has shape {mask.shape}, "
                f"expected ({num_nodes},)."
            )

        boundary_masks[name] = mask

    wall_mask = boundary_masks["wall_mask"]
    inlet_mask = boundary_masks["inlet_mask"]
    outlet_mask = boundary_masks["outlet_mask"]

    _validate_boundary_pair(
        velocity_bc_target,
        velocity_dirichlet_mask,
        (num_samples, num_nodes, 2),
        num_nodes,
        "velocity",
        file_path,
    )

    _validate_boundary_pair(
        pressure_bc_target,
        pressure_dirichlet_mask,
        (num_samples, num_nodes),
        num_nodes,
        "pressure",
        file_path,
    )

    # WLSQ

    neighbors, G_wlsq, cell_index = _process_wlsq(
        neighbors,
        G_wlsq,
        cell_index,
        num_nodes,
        file_path,
    )

    # Node coordinates

    if P.shape[0] == num_nodes:
        pos = P
    elif P.shape[1] == num_nodes:
        pos = P.T
    else:
        raise ValueError(
            f"{file_path}: P has shape {P.shape}, "
            f"but the state has {num_nodes} nodes."
        )

    # Geometry features

    if geometry_features is not None:

        if geometry_features.shape[0] == num_nodes:
            pass
        elif geometry_features.shape[1] == num_nodes:
            geometry_features = geometry_features.T
        else:
            raise ValueError(
                f"{file_path}: geometry_features has shape "
                f"{geometry_features.shape}, incompatible with "
                f"{num_nodes} nodes."
            )

        expected_shape = (
            num_nodes,
            NUM_GEOMETRY_FEATURES,
        )

        if geometry_features.shape != expected_shape:
            raise ValueError(
                f"{file_path}: geometry_features has shape "
                f"{geometry_features.shape}, expected "
                f"{expected_shape}."
            )

        if not np.all(np.isfinite(geometry_features)):
            raise ValueError(
                f"{file_path}: geometry_features contains NaN or Inf."
            )

    # Graph connectivity

    if edge_index.min() < 0 or edge_index.max() >= num_nodes:
        raise ValueError(
            f"{file_path}: edge_index contains invalid node indices."
        )

    # Jacobians

    if jacobian_shared_pattern_raw is None:

        jacobian_shared_pattern = None

    else:

        raw = np.asarray(
            jacobian_shared_pattern_raw
        ).reshape(-1)

        if raw.size != 1:
            raise ValueError(
                f"{file_path}: jacobian_shared_pattern must "
                "contain one value."
            )

        jacobian_shared_pattern = bool(raw[0])

    jacobian_row = _int_vector(jacobian_row)
    jacobian_col = _int_vector(jacobian_col)
    jacobian_shape = _int_vector(jacobian_shape)

    jacobian_dof_p = _int_vector(jacobian_dof_p)
    jacobian_dof_u = _int_vector(jacobian_dof_u)
    jacobian_dof_v = _int_vector(jacobian_dof_v)

    if jacobian_values is not None:

        num_transitions_full = num_snapshots - 1

        if jacobian_values.ndim != 2:
            raise ValueError(
                f"{file_path}: jacobian_values must be 2D."
            )

        if jacobian_values.shape[1] == num_transitions_full:
            jacobian_values = jacobian_values.T
        elif jacobian_values.shape[0] != num_transitions_full:
            raise ValueError(
                f"{file_path}: jacobian_values has shape "
                f"{jacobian_values.shape}; expected one dimension "
                f"to equal {num_transitions_full}."
            )

        jacobian_values = jacobian_values[skip_initial:]

    if jacobian_target_time is not None:
        jacobian_target_time = (
            np.asarray(jacobian_target_time)
            .reshape(-1)[skip_initial:]
        )

    _validate_jacobian(
        file_path=file_path,
        num_nodes=num_nodes,
        num_samples=num_samples,
        t=t,
        skip_initial=skip_initial,
        shared_pattern=jacobian_shared_pattern,
        row=jacobian_row,
        col=jacobian_col,
        values=jacobian_values,
        shape=jacobian_shape,
        dof_p=jacobian_dof_p,
        dof_u=jacobian_dof_u,
        dof_v=jacobian_dof_v,
        target_time=jacobian_target_time,
    )

    return RawDataset(
        X_input=X_input,
        Y_target=Y_target,
        edge_index=edge_index,
        edge_weight=edge_weight,
        delta_t=delta_t,
        pos=pos,

        wall_mask=wall_mask,
        inlet_mask=inlet_mask,
        outlet_mask=outlet_mask,

        neighbors=neighbors,
        G_wlsq=G_wlsq,
        cell_index=cell_index,

        rho=rho,
        mu=mu,
        U_inf=U_inf,
        R=R,
        physics_features=physics_input,
        geometry_features=geometry_features,

        velocity_dirichlet_mask=velocity_dirichlet_mask,
        velocity_bc_target=velocity_bc_target,

        pressure_dirichlet_mask=pressure_dirichlet_mask,
        pressure_bc_target=pressure_bc_target,

        simulation_id=simulation_id,
        file_path=str(file_path),

        jacobian_shared_pattern=jacobian_shared_pattern,
        jacobian_row=jacobian_row,
        jacobian_col=jacobian_col,
        jacobian_values=jacobian_values,
        jacobian_shape=jacobian_shape,

        jacobian_dof_p=jacobian_dof_p,
        jacobian_dof_u=jacobian_dof_u,
        jacobian_dof_v=jacobian_dof_v,

        jacobian_target_time=jacobian_target_time,
    )


def load_simulations(file_paths, skip_initial=0):
    return [
        load_data(
            file_path,
            skip_initial=skip_initial,
            simulation_id=simulation_id,
        )
        for simulation_id, file_path in enumerate(file_paths)
    ]
