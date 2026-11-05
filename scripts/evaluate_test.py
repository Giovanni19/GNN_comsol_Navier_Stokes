#!/usr/bin/env python3

"""
One-step evaluation of a trained run on held-out data, in physical units.

    python scripts/evaluate_test.py outputs/<run>

Everything is read from the run directory and the checkpoints: which
simulations were held out, how the inputs were scaled, and which extra
features each network was fed.
"""

import argparse
import sys
from pathlib import Path
import torch
# Make src/ importable
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from gnn_comsol.evaluation import (
    evaluate,
    read_run,
    report,
    report_boundary,
    report_zero_delta_gain,
    plot_rmse,
    animate_pressure_error,
    animate_velocity_error,
    save_predictions_mat,
    save_evaluation_metrics,
    prepare_evaluation_data,
    build_evaluation_hierarchies,
    load_evaluation_models,
)
# Arguments

def parse_args():

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "run_dir",
        help="a run directory under outputs/"
    )

    parser.add_argument(
        "--simulation-id",
        type=int,
        default=None,
        help=(
            "which held-out simulation to evaluate. Defaults to the "
            "first one the run put in its test block."
        )
    )

    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "evaluate this .mat instead of the one the run held out. "
            "Only for a genuinely unseen file: the default is derived "
            "from the run itself, and is the safe choice."
        )
    )
    parser.add_argument(
        "--skip-initial",
        type=int,
        default=0,
        help=(
            "Number of initial transitions to skip during inference. "
            "This is independent of the skip_initial used during training. "
            "Default: 0, so inference covers the complete trajectory."
        )
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="where to write figures and arrays (default: the run dir)"
    )

    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="skip the pressure-error animation, which is the slow part"
    )

    parser.add_argument(
        "--held-out-only",
        action="store_true",
        help=(
            "evaluate only the samples this run held out, instead of "
            "the whole trajectory. The default covers everything, "
            "because the initial transient is worth looking at and "
            "the MATLAB pipeline wants every transition; the "
            "generalisation numbers are reported apart either way."
        )
    )

    parser.add_argument(
        "--predictions-mat",
        default=None,
        help=(
            "where to write the .mat the MATLAB pipeline reads back "
            "(u_pred, v_pred, p_pred and the fields around them). "
            "Defaults to gnn_predictions.mat inside the output "
            "directory. The MATLAB side passes its own path here, so "
            "that the location is written down in one place instead of "
            "being hard-coded on both sides."
        )
    )

    parser.add_argument(
        "--no-hard-bc",
        action="store_true",
        help=(
            "do not hard-impose the prescribed Dirichlet boundary "
            "conditions on velocity or pressure. By default, velocity "
            "BCs are imposed at wall/inlet nodes and pressure BCs at "
            "pressure-Dirichlet nodes before errors are computed and "
            "before the prediction is written for COMSOL."
        )
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="open the figures in a window as well as saving them"
    )

    return parser.parse_args()


# The evaluation

def main():

    args = parse_args()

    import matplotlib

    if not args.show:
        # plt.show() blocks, which makes an unattended run hang forever
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    run_dir = Path(args.run_dir)

    output_dir = Path(args.output_dir) if args.output_dir else run_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device:  {device}")
    print(f"Run dir: {run_dir}")

    config, metrics = read_run(run_dir)

    print(f"Split:   {metrics['split_mode']}")
    print(f"Held-out simulations: {metrics['test_simulations']}")

    velocity, pressure = load_evaluation_models(
        run_dir,
        device,
    )

    (
        evaluation,
        dataset_path,
        simulation_id,
        time,
        indices,
        report_mask,
        inference_skip_initial,
    ) = prepare_evaluation_data(
        config=config,
        metrics=metrics,
        dataset_override=args.dataset,
        simulation_id=args.simulation_id,
        skip_initial=args.skip_initial,
        held_out_only=args.held_out_only,
    )

    hierarchies = build_evaluation_hierarchies(
        evaluation,
        velocity,
        pressure,
        device,
    )

    # Run it

    (
        rmse,
        pressure_absolute_error,
        velocity_absolute_error,
        pressure_previous_target_error,
        pressure_previous_prediction_error,
        velocity_previous_target_error,
        velocity_previous_prediction_error,
        predictions,
        boundary
    ) = evaluate(
            evaluation,
            velocity,
            pressure,
            hierarchies,
            device,
            hard_bc=not args.no_hard_bc,
    )

    report_boundary(boundary)
    predictions_mat_path = (
        Path(args.predictions_mat)
        if args.predictions_mat
        else output_dir / "gnn_predictions.mat"
    )
    save_predictions_mat(
        predictions_mat_path,
        predictions,
        evaluation,
        time,
        indices,
        inference_skip_initial,
    )

    print(f"u_pred shape: {predictions[:, :, 0].shape}")
    print(f"v_pred shape: {predictions[:, :, 1].shape}")
    print(f"p_pred shape: {predictions[:, :, 2].shape}")

    report(rmse, time, report_mask)
    gains, baseline_errors, gnn_errors = report_zero_delta_gain(
        predictions,
        evaluation.X_input,
        evaluation.Y_target,
        time,
        report_mask,
    )

    save_evaluation_metrics(
        output_dir=output_dir,
        rmse=rmse,
        gains=gains,
        baseline_errors=baseline_errors,
        gnn_errors=gnn_errors,
        pressure_absolute_error=pressure_absolute_error,
        velocity_absolute_error=velocity_absolute_error,
        pressure_previous_target_error=pressure_previous_target_error,
        pressure_previous_prediction_error=pressure_previous_prediction_error,
        velocity_previous_target_error=velocity_previous_target_error,
        velocity_previous_prediction_error=velocity_previous_prediction_error,
        time=time,
        indices=indices,
        simulation_id=simulation_id,
        dataset_path=dataset_path,
        split_mode=metrics["split_mode"],
        report_mask=report_mask,
    )

    plot_rmse(rmse, time, output_dir, report_mask)

    if not args.no_animation:

        print("\nCreating pressure error animation...")
        animate_pressure_error(
            evaluation,
            pressure_absolute_error,
            pressure_previous_target_error,
            time,
            output_dir
        )

        print("\nCreating velocity error animation...")

        animate_velocity_error(
            evaluation,
            velocity_absolute_error,
            velocity_previous_target_error,
            time,
            output_dir
        )

    if args.show:
        plt.show()

    print(f"\nDone. Everything is in {output_dir}")


if __name__ == "__main__":
    main()
