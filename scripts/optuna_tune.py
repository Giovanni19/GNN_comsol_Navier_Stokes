#!/usr/bin/env python3
"""
Hyperparameter optimization with Optuna.

Usage:

    python scripts/optuna_tune.py configs/optuna_velocity.yaml
"""

import argparse
import copy
import sys
from pathlib import Path

import optuna
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnn_comsol.config import load_config
from run_experiment import run_experiment

# ============================================================
# PRUNING SETTINGS
# ============================================================

TIME_WARMUP_EPOCHS = 1
TIME_CHECK_EPOCH = 3

MAX_TRAINING_HOURS = 96.0

PERFORMANCE_PRUNING_START_EPOCH = 50
PERFORMANCE_PRUNING_INTERVAL = 25


def objective(
    trial,
    base_config,
    output_root,
):
    """Run one Optuna trial."""

    config = copy.deepcopy(base_config)

    # ============================================================
    # HYPERPARAMETERS
    # ============================================================

    velocity_lr = trial.suggest_float(
        "velocity_lr",
        1.0e-5,
        1.0e-4,
        log=True,
    )

    velocity_num_neurons = trial.suggest_int(
        "velocity_num_neurons",
        64,
        96,
    )

    velocity_processor_size = trial.suggest_int(
        "velocity_processor_size",
        10,
        15,
    )

    config["networks"]["velocity"]["learning_rate"] = velocity_lr

    config["networks"]["velocity"]["num_neurons"] = (
        velocity_num_neurons
    )

    config["networks"]["velocity"]["processor_size"] = (
        velocity_processor_size
    )

    print(f"\n{'=' * 60}")
    print(f"OPTUNA TRIAL {trial.number}")
    print("=" * 60)

    print(
        f"velocity_lr             = {velocity_lr:.6e}"
    )
    print(
        f"velocity_num_neurons    = {velocity_num_neurons}"
    )
    print(
        f"velocity_processor_size = {velocity_processor_size}"
    )

    best_val_loss = float("inf")
    best_velocity_val_loss = None
    best_pressure_val_loss = None
    best_epoch = None

    epoch_times = []

    def epoch_callback(
        epoch,
        num_epochs,
        train_loss,
        val_loss,
        train_terms,
        val_terms,
        epoch_time,
        velocity_lr,
        pressure_lr,
    ):
        nonlocal best_val_loss
        nonlocal best_velocity_val_loss
        nonlocal best_pressure_val_loss
        nonlocal best_epoch

        if val_loss is None:
            raise RuntimeError(
                "Optuna tuning requires a validation set."
            )

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_velocity_val_loss = val_terms["velocity_data"]
            best_pressure_val_loss = val_terms["pressure_data"]
            best_epoch = epoch

        epoch_times.append(epoch_time)
        trial.report(
            float(val_loss),
            step=epoch - 1,
        )
        print(
            f"Optuna | "
            f"epoch {epoch}/{num_epochs} | "
            f"val_loss={val_loss:.6e} | "
            f"best={best_val_loss:.6e} | "
            f"time={epoch_time:.2f}s"
        )
        # ========================================================
        # COMPUTATIONAL-COST PRUNING
        # ========================================================

        if epoch == TIME_CHECK_EPOCH:

            measured_times = epoch_times[
                TIME_WARMUP_EPOCHS:
            ]

            mean_epoch_time = float(
                np.mean(measured_times)
            )

            estimated_training_hours = (
                mean_epoch_time
                * num_epochs
                / 3600.0
            )

            trial.set_user_attr(
                "mean_epoch_time_seconds",
                mean_epoch_time,
            )

            trial.set_user_attr(
                "estimated_training_hours",
                estimated_training_hours,
            )

            print(
                f"Estimated full training time: "
                f"{estimated_training_hours:.1f} h"
            )

            if (
                estimated_training_hours
                > MAX_TRAINING_HOURS
            ):

                trial.set_user_attr(
                    "prune_reason",
                    "training_too_slow",
                )

                raise optuna.TrialPruned(
                    f"Estimated training time "
                    f"{estimated_training_hours:.1f} h "
                    f"> {MAX_TRAINING_HOURS:.1f} h."
                )
        if (
            epoch >= PERFORMANCE_PRUNING_START_EPOCH
            and
            (
                epoch - PERFORMANCE_PRUNING_START_EPOCH
            ) % PERFORMANCE_PRUNING_INTERVAL == 0
        ):

            if trial.should_prune():

                trial.set_user_attr(
                    "prune_reason",
                    "poor_validation_performance",
                )

                raise optuna.TrialPruned(
                    f"Poor validation performance "
                    f"at epoch {epoch}."
                )
    
    try:

        _, run_dir = run_experiment(
            config=config,
            output_root=output_root,
            verbose=False,
            epoch_callback=epoch_callback,
        )

    except torch.cuda.OutOfMemoryError:

        trial.set_user_attr(
            "prune_reason",
            "cuda_out_of_memory",
        )

        torch.cuda.empty_cache()

        raise optuna.TrialPruned(
            "CUDA out of memory."
        )

    trial.set_user_attr(
        "best_epoch",
        int(best_epoch),
    )

    trial.set_user_attr(
        "best_velocity_val_loss",
        float(best_velocity_val_loss),
    )

    trial.set_user_attr(
        "best_pressure_val_loss",
        float(best_pressure_val_loss),
    )

    trial.set_user_attr(
        "run_dir",
        str(run_dir),
    )

    return float(best_val_loss)


def main():

    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning."
    )

    parser.add_argument(
        "config",
        type=str,
        help="Baseline YAML configuration.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/optuna",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    base_config = load_config(
        args.config
    )

    sampler = optuna.samplers.TPESampler(
        seed=base_config["seed"],
        n_startup_trials=2,
        multivariate=True,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=2,
        n_warmup_steps=49,
        n_min_trials=2,
    )

    study = optuna.create_study(
        study_name="hyperparameter_tuning",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage="sqlite:///optuna.db",
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(
            trial,
            base_config=base_config,
            output_root=args.output_root,
        ),
        n_trials=args.n_trials,
    )


if __name__ == "__main__":
    main()