#!/usr/bin/env python3
"""
Run one experiment, described by a YAML file in configs/.

    python scripts/run_experiment.py configs/coupled_multisimulation_justdata.yaml

Everything a run produces goes into outputs/<name>_<timestamp>/:
the resolved configuration, velocity.pth and pressure.pth, the loss
curve, and the metrics as JSON.

main() at the bottom is the outline of the experiment; every step above
it is one function.
"""


import sys
from pathlib import Path

import torch

# Make the package importable without installing it first
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_comsol import data as gdata
from gnn_comsol.config import load_config
from gnn_comsol.experiment import load_simulations, split_simulations
from gnn_comsol.graph.hierarchy import build_bsms_hierarchies
from gnn_comsol.outputs import create_run_dir, write_outputs
from gnn_comsol.physics import (
    build_physics_geometries,
    compute_physics_reference_scales,
)
from gnn_comsol.training import run_coupled_training
from gnn_comsol.utils import parse_args, set_seeds

# The experiment
def run_experiment(
    config,
    output_root,
    verbose=True,
    epoch_callback=None,
):
    """Run one complete training experiment."""

    set_seeds(config["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    run_dir = create_run_dir(
        output_root,
        config,
    )

    print(f"Device:  {device}")
    print(f"Run dir: {run_dir}")

    simulations = load_simulations(config)

    state_scales = gdata.build_state_scales(
        simulations
    )

    splits = split_simulations(
        simulations,
        config,
    )

    (
        physics_normalizer,
        geometry_normalizer,
        dt_mean,
        dt_std,
    ) = gdata.build_normalizers(
        splits.train,
        config,
    )

    print(
        f"delta_t normalization: "
        f"mean={dt_mean:.6e}, std={dt_std:.6e}"
    )

    if physics_normalizer is not None:
        print(f"\n{physics_normalizer}")

    if geometry_normalizer is not None:
        print(f"\n{geometry_normalizer}")

    normalized = gdata.normalize_splits(
        splits,
        physics_normalizer,
        geometry_normalizer,
        dt_mean,
        config,
    )

    bsms_hierarchies = build_bsms_hierarchies(
        simulations,
        config,
    )

    physics_geometries = build_physics_geometries(
        simulations,
        config.get("fluid"),
    )

    (
        continuity_reference,
        momentum_reference,
    ) = compute_physics_reference_scales(
        splits.train,
        physics_geometries,
    )

    graph_topologies = gdata.build_graph_topologies(
        simulations
    )

    coupled_loaders = gdata.build_coupled_loaders(
        normalized,
        config,
    )

    trained = run_coupled_training(
        config=config,
        coupled_loaders=coupled_loaders,
        graph_topologies=graph_topologies,
        bsms_hierarchies=bsms_hierarchies,
        physics_geometries=physics_geometries,
        state_scales=state_scales,
        physics_normalizer=physics_normalizer,
        geometry_normalizer=geometry_normalizer,
        continuity_reference=continuity_reference,
        momentum_reference=momentum_reference,
        dt_mean=dt_mean,
        dt_std=dt_std,
        run_dir=run_dir,
        device=device,
        verbose=verbose,
        epoch_callback=epoch_callback,
    )

    write_outputs(
        run_dir,
        config,
        splits,
        trained,
    )

    return trained, run_dir


def main():

    args = parse_args()

    config = load_config(
        args.config
    )

    trained, run_dir = run_experiment(
        config=config,
        output_root=args.output_root,
        verbose=not args.quiet,
    )

    print(
        f"\nDone. Everything is in {run_dir}"
    )


if __name__ == "__main__":
    main()
