"""
Saving evaluation results to disk.

This module writes:
- GNN predictions and reference fields to MATLAB .mat format;
- evaluation metrics and diagnostics to NumPy .npz format.
"""

from pathlib import Path

import numpy as np
from scipy.io import savemat


def save_predictions_mat(
    path,
    predictions,
    evaluation,
    time,
    indices,
    inference_skip_initial,
):
    """
    Save GNN predictions and the corresponding COMSOL data for MATLAB.

    The predictions are stored in physical units together with:
    - the GNN input state;
    - the COMSOL target state;
    - timestep information;
    - mesh coordinates;
    - Python and COMSOL transition indices.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    savemat(
        path,
        {
            # GNN predictions in physical units
            "u_pred": predictions[:, :, 0],
            "v_pred": predictions[:, :, 1],
            "p_pred": predictions[:, :, 2],

            # State given as input to the GNN
            "u_input": evaluation.X_input[:, :, 0],
            "v_input": evaluation.X_input[:, :, 1],
            "p_input": evaluation.X_input[:, :, 2],

            # True COMSOL target
            "u_target": evaluation.Y_target[:, :, 0],
            "v_target": evaluation.Y_target[:, :, 1],
            "p_target": evaluation.Y_target[:, :, 2],

            # Time information
            "delta_t": np.asarray(
                evaluation.delta_t
            ).reshape(-1, 1),

            "target_time": np.asarray(
                time
            ).reshape(-1, 1),

            # Mesh-node coordinates
            "node_coordinates": np.asarray(
                evaluation.pos
            ),

            # Python sample index
            "sample_index_python": np.asarray(
                indices
            ).reshape(-1, 1),

            # Original COMSOL transition index
            "transition_index": (
                np.asarray(indices)
                + inference_skip_initial
                + 1
            ).reshape(-1, 1),
        },
    )

    print(f"\nGNN predictions saved to: {path}")


def save_evaluation_metrics(
    output_dir,
    rmse,
    gains,
    baseline_errors,
    gnn_errors,
    pressure_absolute_error,
    velocity_absolute_error,
    pressure_previous_target_error,
    pressure_previous_prediction_error,
    velocity_previous_target_error,
    velocity_previous_prediction_error,
    time,
    indices,
    simulation_id,
    dataset_path,
    split_mode,
    report_mask,
):
    """
    Save evaluation errors, zero-delta comparison and RMSE histories.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_name = (
        "test_rmse_over_time.npz"
        if report_mask is None
        else "trajectory_rmse_over_time.npz"
    )

    output_path = output_dir / file_name

    held_out_mask = (
        np.ones(len(time), dtype=bool)
        if report_mask is None
        else report_mask
    )

    np.savez(
        output_path,

        time=time,
        indices=indices,
        simulation_id=simulation_id,
        dataset=str(dataset_path),
        split_mode=split_mode,
        held_out_mask=held_out_mask,

        # Gain over zero-delta baseline
        gain_u=gains["u"],
        gain_v=gains["v"],
        gain_velocity=gains["velocity"],
        gain_p=gains["p"],

        # Zero-delta errors
        baseline_rmse_u=baseline_errors["u"],
        baseline_rmse_v=baseline_errors["v"],
        baseline_rmse_velocity=baseline_errors["velocity"],
        baseline_rmse_p=baseline_errors["p"],

        # GNN errors
        gnn_rmse_u=gnn_errors["u"],
        gnn_rmse_v=gnn_errors["v"],
        gnn_rmse_velocity=gnn_errors["velocity"],
        gnn_rmse_p=gnn_errors["p"],

        # Spatial error fields
        pressure_absolute_error=pressure_absolute_error,
        velocity_absolute_error=velocity_absolute_error,

        pressure_previous_target_error=(
            pressure_previous_target_error
        ),

        pressure_previous_prediction_error=(
            pressure_previous_prediction_error
        ),

        velocity_previous_target_error=(
            velocity_previous_target_error
        ),

        velocity_previous_prediction_error=(
            velocity_previous_prediction_error
        ),

        # Per-variable RMSE
        **{
            f"rmse_{name}": values
            for name, values in rmse.items()
        },
    )

    print(f"Evaluation metrics saved to: {output_path}")

    return output_path