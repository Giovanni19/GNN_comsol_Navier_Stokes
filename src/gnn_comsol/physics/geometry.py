"""
The mesh-static quantities the physics loss needs, one per simulation.

They depend on the mesh alone, so the WLSQ operators and the control
volumes are built once and cached per (device, dtype).
"""

import torch
from .fluid import FluidProperties
from .mesh import build_cell_geometry
from .wlsq import build_wlsq_operators


class PhysicsGeometry:
    """
    WLSQ operators and control volumes of one simulation.

    Parameters
    ----------
    neighbors : list[np.ndarray]
        Zero-based WLSQ stencil of every node.

    G_wlsq : list[np.ndarray]
        Per-node (2, k_i) gradient operators.

    cell_index : np.ndarray, shape (Nc, 3)
        Node indices of each triangular cell.

    pos : np.ndarray, shape (N, 2)
        Node coordinates.

    num_nodes : int

    fluid : FluidProperties or None
        Needed by the momentum residual only.

    simulation_id : int, optional
        Only used in error messages.
    """

    def __init__(
        self,
        neighbors,
        G_wlsq,
        cell_index,
        pos,
        num_nodes,
        fluid=None,
        simulation_id=None,
    ):

        self.neighbors = neighbors
        self.G_wlsq = G_wlsq
        self.raw_cell_index = cell_index
        self.pos = pos
        self.num_nodes = num_nodes
        self.fluid = fluid
        self.simulation_id = simulation_id

        self._operators = {}
        self._cells = {}

    @classmethod
    def from_simulation(cls, simulation, fluid=None):
        """
        Build from a RawDataset, or None if it carries no WLSQ data
        (older .mat files).
        """

        has_wlsq = (
            simulation.neighbors is not None
            and simulation.G_wlsq is not None
            and simulation.cell_index is not None
        )

        if not has_wlsq:
            return None

        return cls(
            neighbors=simulation.neighbors,
            G_wlsq=simulation.G_wlsq,
            cell_index=simulation.cell_index,
            pos=simulation.pos,
            num_nodes=simulation.num_nodes,
            fluid=fluid,
            simulation_id=simulation.simulation_id,
        )

    def operators(self, device=None, dtype=torch.float32):
        """The WLSQ operators on `device`, built on first use."""

        key = (str(device), dtype)

        if key not in self._operators:

            self._operators[key] = build_wlsq_operators(
                self.neighbors,
                self.G_wlsq,
                num_nodes=self.num_nodes,
                device=device,
                dtype=dtype,
            )

        return self._operators[key]

    def cells(self, device=None, dtype=torch.float32):
        """The control volumes on `device`, built on first use."""

        key = (str(device), dtype)

        if key not in self._cells:

            self._cells[key] = build_cell_geometry(
                self.pos,
                self.raw_cell_index,
                device=device,
                dtype=dtype,
            )

        return self._cells[key]

    def require_fluid(self):
        """
        The fluid properties, or a refusal that says how to supply them.
        """

        if self.fluid is None:
            raise ValueError(
                f"Simulation {self.simulation_id} has no fluid properties."
            )

        return self.fluid

    @property
    def num_cells(self):
        return self.raw_cell_index.shape[0]

    def __repr__(self):

        return (
            f"PhysicsGeometry(simulation_id={self.simulation_id}, "
            f"num_nodes={self.num_nodes}, "
            f"num_cells={self.num_cells}, "
            f"fluid={self.fluid})"
        )


def build_physics_geometries(simulations, fluid_config=None):
    """
    Build the PhysicsGeometry object for every simulation.
    """

    physics_geometries = {}

    for simulation in simulations:

        fluid = FluidProperties.from_simulation(simulation)

        if fluid is None:
            fluid = FluidProperties.from_config(
                fluid_config,
                simulation.simulation_id,
            )

        geometry = PhysicsGeometry.from_simulation(
            simulation,
            fluid=fluid,
        )

        if geometry is not None:
            physics_geometries[simulation.simulation_id] = geometry

    return physics_geometries