# GNN-COMSOL

**Graph Neural Networks for Accelerating Nonlinear Navier–Stokes Solvers in COMSOL Multiphysics**

This repository contains a machine learning implementation designed to investigate the use of Graph Neural Networks (GNNs) as data-driven initial guesses for nonlinear finite-element solvers.

The framework combines **COMSOL Multiphysics, MATLAB, Python, PyTorch Geometric and NVIDIA PhysicsNemo** to train mesh-based neural networks and evaluate their impact on Newton solver convergence.

---

## Overview

The project investigates whether Graph Neural Networks can accelerate time-dependent incompressible Navier–Stokes simulations by providing improved initial guesses to the nonlinear solver.

Instead of replacing the numerical solver, the proposed approach uses machine learning to predict the solution at the next time step.

These predictions are then imported into COMSOL Multiphysics and used to initialize the corresponding nonlinear finite-element problem.

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
- (Physics-based quantities derived from the flow fields)

The networks can be trained to predict nondimensional solution increments rather than absolute solution values.

## Governing Equations

The considered physical problem is the two-dimensional incompressible Navier–Stokes system.
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

# Installation and Usage

This section describes how to install the framework, generate the training data, train the Graph Neural Networks, and evaluate their performance within COMSOL Multiphysics.

### 1. Prerequisites

Before running the framework, make sure the following software is available:

- **COMSOL Multiphysics** with the CFD Module and LiveLink for MATLAB.
- **MATLAB**, with a version compatible with the installed COMSOL release.
- **Python**, with a version compatible with the dependencies specified in `requirements.txt`.
- **PyTorch** and **PyTorch Geometric**.
- **NVIDIA PhysicsNeMo**, for the BSMS architecture.

An NVIDIA GPU with CUDA support is recommended for training, although the framework can also run on a CPU.

### 2. Clone the Repository

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/Giovanni19/GNN_comsol_Navier_Stokes.git
cd GNN_comsol_Navier_Stokes
```

### 3. Set Up the Python Environment

It is recommended to use a dedicated Python virtual environment to avoid dependency conflicts.

**Windows (PowerShell)**

Create the environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If CUDA acceleration is required, ensure that the installed PyTorch build is compatible with the NVIDIA GPU and driver.

The exact CUDA-enabled PyTorch installation instructions are available in the [official PyTorch documentation](https://pytorch.org/get-started/locally/).

### 4. Generate the COMSOL Database

The first step consists of generating the reference simulation data used to train the GNN models.

**Step 1 — Create and run the COMSOL model**

Implement the desired fluid dynamics model in COMSOL Multiphysics, configure the time-dependent study, run the simulation, and save the resulting `.mph` file.

The current framework is designed for two-dimensional incompressible Navier–Stokes simulations.

**Step 2 — Start LiveLink for MATLAB**

Launch MATLAB through **COMSOL with MATLAB / LiveLink for MATLAB**, rather than opening a standalone MATLAB session.

This ensures that the COMSOL API and the connection to the COMSOL server are properly initialized.

**Step 3 — Configure the data extraction**

Add the MATLAB scripts located in:

```text
src/gnn_comsol/Matlab/
```

to the MATLAB working directory or search path.

Open:

```matlab
first_database.m
```

Configure the `get_config()` function according to the COMSOL model, including the relevant model file, solution dataset, and export settings.

**Step 4 — Generate the dataset**

Execute:

```matlab
first_database
```

The script generates a `.mat` database containing the information required to construct the graph dataset, including solution fields, mesh information, and the relevant geometric and boundary-condition data.

The generated database will subsequently be used for GNN training.

### 5. Configure the Neural Networks

Training experiments are configured through YAML files located in:

```text
configs/
```

Several example configurations are provided in the repository.

Create a new configuration file, for example:

```text
configs/my_experiment.yaml
```

Configure the relevant parameters, including:

- Paths to the generated `.mat` databases.
- Velocity and pressure network architectures.
- Network depth and hidden dimensions.
- Learning rates and training epochs.
- Training, validation, and test splits.
- Input features and normalization.
- Data-driven, physics-based, and Jacobian-based loss settings.

Make sure the dataset paths correspond to the databases generated in the previous step.

### 6. Train the Models

From the root directory of the repository, with the Python environment activated, run:

```bash
python scripts/run_experiment.py configs/my_experiment.yaml
```

The script loads the simulation databases, constructs the graph representations, initializes the velocity and pressure networks, and performs coupled training.

The experiment results are saved in a timestamped directory under:

```text
outputs/
```

The generated files include:

- `velocity.pth` — Trained velocity network checkpoint.
- `pressure.pth` — Trained pressure network checkpoint.
- Training and validation loss histories.
- Experiment configuration and evaluation metrics.

When a validation dataset is available, the checkpoints correspond to the epoch with the lowest combined validation loss. Otherwise, the final training epoch is retained.

### 7. Evaluate the Trained Models

The trained networks can be evaluated on simulation data using:

```text
scripts/evaluate_test.py
```

This allows the predicted velocity and pressure fields to be compared with the reference COMSOL solutions before performing the nonlinear solver evaluation.

The evaluation can be used to investigate prediction accuracy, generalization across simulation conditions, and the quality of the generated initial guesses.

### 8. Evaluate Newton Solver Convergence in COMSOL

The final step consists of importing the trained GNN predictions into COMSOL and comparing nonlinear solver convergence against the standard initialization.

**Step 1 — Open COMSOL with MATLAB**

Launch COMSOL with LiveLink for MATLAB and load the corresponding COMSOL model.

**Step 2 — Configure the experiment**

Open:

```matlab
gnn_comsol_config.m
```

Configure the experiment settings, including the paths to the trained model checkpoints, the relevant simulation data, and the transitions to be evaluated.

**Step 3 — Run the complete pipeline**

Execute:

```matlab
complete_pipeline_all_timesteps
```

Select the desired number of time transitions to evaluate.

The pipeline uses the trained GNNs to generate initial guesses for the nonlinear stationary problems corresponding to the selected time steps.

**Step 4 — Compare solver performance**

The evaluation compares:

- Newton iterations using the standard COMSOL initialization.
- Newton iterations using the GNN-predicted initial guess.
- Differences in convergence behavior between the two initialization strategies.

The objective is to assess whether the learned initial guesses provide a measurable improvement in nonlinear solver convergence.

---

### Important Notes

- The COMSOL `.mph` models and generated `.mat` datasets must be provided or generated separately unless explicitly included in the repository.
- MATLAB and COMSOL versions must be compatible with the installed LiveLink interface.
- Training on a CUDA-enabled GPU is recommended for larger meshes and datasets.
- The current framework is primarily designed for two-dimensional incompressible laminar flow simulations.
- Improved prediction accuracy does not necessarily guarantee fewer Newton iterations. The solver evaluation is therefore an essential part of the workflow.
