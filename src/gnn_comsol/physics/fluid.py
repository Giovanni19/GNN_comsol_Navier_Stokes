"""
Fluid properties required by the momentum residual.

They come from the .mat when first_database.m could read `spf.rho`
and `spf.mu` from the COMSOL model, and from the `fluid:` section
of the experiment YAML otherwise.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FluidProperties:
    """Density and dynamic viscosity in SI units."""

    rho: float
    mu: float

    def __post_init__(self):
        if self.rho <= 0 or self.mu <= 0:
            raise ValueError(
                "rho and mu must be positive."
            )

    @classmethod
    def from_simulation(cls, simulation):
        """Build from the .mat, or return None if unavailable."""

        if simulation.rho is None or simulation.mu is None:
            return None

        return cls(
            rho=float(simulation.rho),
            mu=float(simulation.mu),
        )

    @classmethod
    def from_config(cls, entry, simulation_id=None):
        """Build from the `fluid:` section of the experiment YAML."""

        if entry is None:
            return None

        # One pair shared by all simulations.
        if "rho" in entry and "mu" in entry:
            return cls(
                rho=float(entry["rho"]),
                mu=float(entry["mu"]),
            )

        # One pair for each simulation.
        if simulation_id is None:
            return None

        per_simulation = entry.get(simulation_id)

        if per_simulation is None:
            return None

        return cls(
            rho=float(per_simulation["rho"]),
            mu=float(per_simulation["mu"]),
        )