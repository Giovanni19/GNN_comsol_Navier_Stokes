# GNN-COMSOL

**Graph Neural Networks for Accelerating Nonlinear Navier–Stokes Solvers in COMSOL Multiphysics**

This repository contains a machine learning framework designed to investigate the use of Graph Neural Networks (GNNs) as data-driven initial guesses for nonlinear finite-element solvers.

The framework combines **COMSOL Multiphysics, MATLAB, PyTorch, and PyTorch Geometric** to train mesh-based neural networks and evaluate their impact on Newton solver convergence.

---

## Overview

The project investigates whether Graph Neural Networks can accelerate time-dependent incompressible Navier–Stokes simulations by providing improved initial guesses to the nonlinear solver.

Instead of replacing the numerical solver, the proposed approach uses machine learning to predict the solution at the next time step:

$$
(\mathbf{u}^{n},p^{n})
\longrightarrow
(\mathbf{u}^{n+1},p^{n+1}).
$$

These predictions are subsequently imported into COMSOL Multiphysics and used to initialize the corresponding nonlinear finite-element problem.

The primary objective is to evaluate whether improved solution predictions translate into **fewer Newton iterations and reduced computational cost**.

## Neural Network Architectures

The framework employs two separate Graph Neural Networks, trained jointly to predict the velocity and pressure fields.

| Field | Architecture | Description |
|---|---|---|
| Velocity $(u,v)$ | **MeshGraphNet (MGN)** | Message-passing neural network operating on unstructured meshes |
| Pressure $p$ | **Bi-Stride Multi-Scale GNN (BSMS)** | Multi-resolution graph architecture designed to capture long-range spatial dependencies |

Both networks operate directly on the finite-element mesh exported from COMSOL.

The input features can include:

- Previous velocity and pressure fields
- Time-step information
- Mesh geometry and connectivity
- Boundary-condition and node-type information
- Physics-based quantities derived from the flow fields

The networks can be trained to predict nondimensional solution increments rather than absolute solution values.

## Governing Equations

The considered physical problem is the two-dimensional incompressible Navier–Stokes system.

**Continuity equation**

$$
\nabla\cdot\mathbf{u}=0
$$

**Momentum equation**

$$
\rho\left(
\frac{\partial\mathbf{u}}{\partial t}
+(\mathbf{u}\cdot\nabla)\mathbf{u}
\right)
=
-\nabla p+\mu\nabla^2\mathbf{u}
$$

where:

- $\mathbf{u}=(u,v)$ is the velocity field.
- $p$ is the pressure.
- $\rho$ is the fluid density.
- $\mu$ is the dynamic viscosity.

The numerical experiments consider laminar flow through a two-dimensional channel containing a cylindrical obstacle, across different Reynolds numbers and geometric configurations.

## Training Objectives

The framework supports several training objectives.

**Data-driven loss**

Minimizes the discrepancy between the predicted and reference COMSOL solutions.

**Physics-informed losses**

Incorporates continuity and momentum residuals to encourage physically consistent predictions.

**Jacobian-based loss**

Uses the finite-element Jacobian exported from COMSOL to construct a linearized residual-based training objective.

The combined training objective can be expressed as:

$$
\mathcal{L}=
\lambda_u\mathcal{L}_u+
\lambda_p\mathcal{L}_p+
\lambda_c\mathcal{L}_{continuity}+
\lambda_m\mathcal{L}_{momentum}+
\lambda_N\mathcal{L}_{Newton}.
$$

The framework also supports hyperparameter optimization and systematic parameter sweeps.

## Computational Workflow

The complete pipeline integrates COMSOL, MATLAB, and Python.

1. **Data generation:** Run time-dependent Navier–Stokes simulations in COMSOL Multiphysics.
2. **Data extraction:** Export solution fields, mesh connectivity, boundary conditions, and geometric information through MATLAB.
3. **Graph construction:** Convert the finite-element mesh and simulation data into graph representations.
4. **Model training:** Train the coupled velocity and pressure networks using PyTorch.
5. **Flow prediction:** Predict the velocity and pressure fields at the next time step.
6. **Solver initialization:** Import the predicted fields into COMSOL as initial guesses.
7. **Performance evaluation:** Compare Newton iterations and convergence behavior against the standard solver initialization.

## Repository Structure

```text
GNN_comsol_Navier_Stokes/
├── configs/          # Experiment and training configurations
├── docs/             # Documentation and supporting files
├── scripts/          # Training, evaluation and tuning scripts
├── src/
│   └── gnn_comsol/
│       ├── Matlab/   # COMSOL–MATLAB integration
│       ├── data/     # Data processing and graph construction
│       ├── graph/    # Graph hierarchy utilities
│       ├── models/   # GNN architectures
│       ├── physics/  # Physical residuals and mesh operators
│       ├── training/ # Training loops and loss functions
│       └── evaluation/
└── tests/            # Unit tests and validation utilities
```

Experiment outputs, trained checkpoints, and evaluation results are generated locally during execution.

## Technologies

- **COMSOL Multiphysics** — Finite-element simulations and nonlinear solver evaluation
- **MATLAB / LiveLink for MATLAB** — Simulation automation and FEM data extraction
- **Python** — Machine learning pipeline and data processing
- **PyTorch** — Neural network training and automatic differentiation
- **PyTorch Geometric** — Graph neural network operations
- **NVIDIA PhysicsNeMo** — Multi-scale graph neural network components
- **Optuna** — Hyperparameter optimization

## Research Objective

The central research question is:

**Can Graph Neural Networks provide initial guesses that improve the convergence of Newton-based nonlinear solvers for time-dependent fluid dynamics simulations?**

The project investigates the relationship between prediction accuracy, physics-based training objectives, and nonlinear solver convergence.

A particular focus is placed on understanding why improvements in solution prediction accuracy do not necessarily lead to reductions in Newton iterations, and whether solver-aware learning objectives can help bridge this gap.
