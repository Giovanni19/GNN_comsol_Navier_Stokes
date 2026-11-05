"""
GNN surrogate for a 2D channel flow solved in COMSOL.

The mesh is read as a graph: nodes carry the local state (u, v, p),
edges carry the mesh adjacency, and the networks learn the one-step
time-advance operator

    X^k  ->  X^(k+1)

Layout
------
    data/         reading, splitting, scaling, feature encodings
    models/       network architectures
    graph/        bi-stride graph hierarchy for the BSMS model
    physics/      WLSQ gradients and Navier-Stokes residuals
    boundary.py   hard imposition of the Dirichlet boundary conditions
    train.py      the coupled training loop
    checkpoints.py  saving, with the feature scalings attached
    config.py     experiment configuration loading and validation
    plots.py      figures

Experiments live in configs/ and are run with
scripts/run_experiment.py.
"""

__version__ = "0.1.0"
