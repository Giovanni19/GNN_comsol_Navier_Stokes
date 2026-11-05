# GNN-COMSOL

Graph Neural Network surrogates for accelerating incompressible Navier–Stokes simulations in COMSOL Multiphysics.

## Overview

This project investigates the use of Graph Neural Networks (GNNs) to predict the evolution of fluid-flow solutions on unstructured finite-element meshes.

The models are trained on time-dependent COMSOL simulations to predict the next flow state:

$$
(u^n, v^n, p^n) \rightarrow (u^{n+1}, v^{n+1}, p^{n+1}).
$$

The predicted solution is then used as an initial guess for the nonlinear solver in COMSOL, with the goal of reducing the number of Newton iterations required for convergence.

## Models

The framework uses separate networks for velocity and pressure:

- **Velocity:** MeshGraphNet
- **Pressure:** Bi-Stride Multi-Scale Graph Neural Network (BSMS)

The models operate directly on the unstructured COMSOL mesh and can include flow, geometric, boundary-condition, and physics-based features.

## Physics

The considered problem is the two-dimensional incompressible Navier–Stokes system:

$$
\nabla \cdot \mathbf{u} = 0,
$$

$$
\rho
\left(
\frac{\partial \mathbf{u}}{\partial t}
+
(\mathbf{u}\cdot\nabla)\mathbf{u}
\right)
=
-\nabla p
+
\mu \nabla^2\mathbf{u}.
$$

Physics-based losses and FEM/Jacobian-based diagnostics can be included during training and evaluation.

## Workflow

The main pipeline is:

1. Generate time-dependent simulations in COMSOL.
2. Export the FEM mesh, solution fields, boundary information, and geometric features.
3. Train the GNN models.
4. Predict the next flow state.
5. Import the prediction into COMSOL.
6. Use it as an initial guess for the corresponding nonlinear stationary solve.
7. Compare convergence against the standard COMSOL initial guess.

## Repository Structure

```text
configs/        Training and experiment configurations
scripts/        Training, evaluation, and utility scripts
src/            Core GNN-COMSOL implementation
tests/          Tests and validation scripts
outputs/        Trained models and experiment results
```

## Main Technologies

- Python
- PyTorch
- PyTorch Geometric
- NVIDIA PhysicsNeMo
- MATLAB
- COMSOL Multiphysics / LiveLink for MATLAB

## Goal

The main objective is to determine whether learned flow predictions can provide better initial guesses for nonlinear FEM solvers and therefore reduce the computational cost of solving time-dependent fluid dynamics problems.