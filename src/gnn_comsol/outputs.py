"""
Utilities for writing experiment outputs and summary metrics.
"""

import json

import numpy as np
from datetime import datetime
import json
from pathlib import Path


def create_run_dir(output_root, config):
    """Create the run directory and save the resolved configuration."""

    run_dir = Path(output_root) / (
        f"{config['name']}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as handle:
        json.dump(config, handle, indent=2)

    return run_dir

def write_outputs(run_dir, config, splits, trained):
    """Write the summary metrics of a completed experiment."""

    training_summary = trained.get("_training_summary", {})

    train_history = training_summary.get(
        "train_history",
        [],
    )

    val_history = training_summary.get(
        "val_history",
        [],
    )

    if train_history:
        final_train_loss = float(train_history[-1])
    else:
        final_train_loss = None

    if val_history:

        best_epoch_index = int(np.argmin(val_history))
        best_epoch = best_epoch_index + 1

        best_val_loss = float(
            val_history[best_epoch_index]
        )

        final_val_loss = float(
            val_history[-1]
        )

        train_loss_at_best_val = float(
            train_history[best_epoch_index]
        )

    else:

        best_epoch = None
        best_val_loss = None
        final_val_loss = None
        train_loss_at_best_val = None

    results = {
        "experiment": config["name"],

        "split_mode": config["split"]["mode"],

        "train_simulations": [
            sim.simulation_id
            for sim in splits.train
        ],

        "val_simulations": [
            sim.simulation_id
            for sim in splits.val
        ],

        "test_simulations": [
            sim.simulation_id
            for sim in splits.test
        ],

        "samples": {
            name: sum(
                sim.num_samples
                for sim in getattr(splits, name)
            )
            for name in ("train", "val", "test")
        },

        "training": {
            "num_epochs": config["training"]["num_epochs"],
            "final_train_loss": final_train_loss,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "final_val_loss": final_val_loss,
            "train_loss_at_best_val": train_loss_at_best_val,
        },
    }

    with open(run_dir / "metrics.json", "w") as handle:
        json.dump(results, handle, indent=2)