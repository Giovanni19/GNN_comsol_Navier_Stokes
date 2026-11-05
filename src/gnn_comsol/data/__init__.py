"""Loading, splitting, scaling and graph construction."""

from .features import (
    FEATURE_SIZES,
    NODE_TYPE_FEATURE_NAMES,
    NUM_NODE_TYPE_FEATURES,
    add_time_feature,
    add_time_fourier_features,
    build_features,
    build_node_type_features,
    features_size,
)

from .graphs import (
    build_graph_topologies,
    create_coupled_dataset,
    to_tensor,
)

from .loading import (
    RawDataset,
    load_data,
    load_simulations
)

from .normalization import (
    PRESSURE_COLUMNS,
    STATE_COLUMNS,
    TARGET_COLUMNS,
    VARIABLE_NAMES,
    VELOCITY_COLUMNS,
    normalize_simulation,
    PHYSICS_FEATURE_NAMES,
    NUM_PHYSICS_FEATURES,
    PhysicsNormalizer,
    compute_multi_simulation_physics_normalization_parameters,
    compute_multi_simulation_geometry_normalization_parameters,
    GEOMETRY_FEATURE_NAMES,
    NUM_GEOMETRY_FEATURES,
    GeometryNormalizer,
    build_normalizers,
    normalize_splits,
    build_state_scales,
)


from .splitting import (
    SimulationSplits,
    compute_split_indices,
    split_simulations,
    split_simulations_by_sample,
    subset_simulation
)
from .loaders import build_coupled_loaders

__all__ = [
    "FEATURE_SIZES",
    "PRESSURE_COLUMNS",
    "RawDataset",
    "STATE_COLUMNS",
    "SimulationSplits",
    "TARGET_COLUMNS",
    "VARIABLE_NAMES",
    "VELOCITY_COLUMNS",
    "add_time_feature",
    "add_time_fourier_features",
    "build_features",
    "build_graph_topologies",
    "build_coupled_loaders",
    "build_normalizers",
    "build_state_scales",
    "normalize_splits",
    "features_size",
    "compute_multi_simulation_geometry_normalization_parameters",
    "compute_split_indices",
    "create_coupled_dataset",
    "load_data",
    "normalize_simulation",
    "load_simulations",
    "split_simulations",
    "split_simulations_by_sample",
    "subset_simulation",
    "to_tensor",
    "NUM_PHYSICS_FEATURES",
    "PHYSICS_FEATURE_NAMES",
    "PhysicsNormalizer",
    "compute_multi_simulation_physics_normalization_parameters",
    "NUM_GEOMETRY_FEATURES",
    "GEOMETRY_FEATURE_NAMES",
    "GeometryNormalizer",
    "compute_multi_simulation_geometry_normalization_parameters",
]
