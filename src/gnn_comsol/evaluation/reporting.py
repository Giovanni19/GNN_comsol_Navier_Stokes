"""
Reporting utilities for model evaluation.
"""

import numpy as np


def report(rmse, time, held_out_mask=None):
    """
    RMSE per variable, in physical units.

    held_out_mask
        True where the sample was NOT used for training. Numbers over
        the whole trajectory are a diagnostic; the generalisation figure
        is the one restricted to the mask, reported apart. None when
        every sample is held out.
    """

    everything_held_out = (
        held_out_mask is None or bool(held_out_mask.all())
    )

    print("\n" + "=" * 60)

    if everything_held_out:
        print("HELD-OUT RMSE - PHYSICAL UNITS")
    else:
        print("RMSE OVER THE WHOLE TRAJECTORY - PHYSICAL UNITS")
        print("(diagnostic: most of these samples WERE trained on)")

    print("=" * 60)

    for name, values in rmse.items():

        worst = int(np.argmax(values))

        print(f"\n{name}")
        print(f"  Mean RMSE:     {values.mean():.6e}")
        print(f"  Median RMSE:   {np.median(values):.6e}")
        print(f"  Min RMSE:      {values.min():.6e}")
        print(f"  Max RMSE:      {values.max():.6e}")
        print(f"  Max at sample: {worst}")
        print(f"  Max at time:   {time[worst]:.6e}")

    if everything_held_out:
        return

    # The number that actually measures generalisation
    print("\n" + "=" * 60)
    print(
        f"HELD-OUT RMSE - PHYSICAL UNITS "
        f"({int(held_out_mask.sum())} of {len(held_out_mask)} samples)"
    )
    print("=" * 60)

    if not held_out_mask.any():
        print("\n  No sample was held out by this run's split.")
        return

    for name, values in rmse.items():

        held_out_values = values[held_out_mask]

        worst = int(np.argmax(held_out_values))

        worst_sample = int(np.flatnonzero(held_out_mask)[worst])

        print(f"\n{name}")
        print(f"  Mean RMSE:     {held_out_values.mean():.6e}")
        print(f"  Median RMSE:   {np.median(held_out_values):.6e}")
        print(f"  Min RMSE:      {held_out_values.min():.6e}")
        print(f"  Max RMSE:      {held_out_values.max():.6e}")
        print(f"  Max at sample: {worst_sample}")
        print(f"  Max at time:   {time[worst_sample]:.6e}")


def report_boundary(boundary):
    """What the hard imposition of the boundary conditions did."""

    if boundary is None:
        return

    print("\n========================================")
    print("HARD BOUNDARY CONDITIONS")
    print("========================================")

    print(
        f"Dirichlet nodes : {boundary['num_nodes']} "
        f"of {boundary['num_total']} "
        f"({100 * boundary['num_nodes'] / boundary['num_total']:.1f}%)"
    )

    print(
        "Prediction error there, BEFORE the values were imposed:\n"
        f"  RMS over samples : {boundary['rms'].mean():.6e}\n"
        f"  worst sample RMS : {boundary['rms'].max():.6e}\n"
        f"  worst node       : {boundary['max'].max():.6e}"
    )

    print(
        "\nThose nodes now carry the prescribed value exactly, so the "
        "reported RMSE and the field written for COMSOL both satisfy "
        "the boundary conditions."
    )


def report_zero_delta_gain(
    predictions,
    previous,
    target,
    time,
    held_out_mask=None,
):
    """
    Compare the GNN against the zero-delta predictor Y_pred = X_k.

        gain = RMSE(previous, target) / RMSE(GNN, target)

    gain > 1 means the GNN is better.
    """

    eps = 1e-15

    def scalar_rmse(a, b):
        return np.sqrt(np.mean((a - b) ** 2, axis=1))

    # Per-sample, over all mesh nodes
    baseline_u = scalar_rmse(previous[:, :, 0], target[:, :, 0])
    baseline_v = scalar_rmse(previous[:, :, 1], target[:, :, 1])
    baseline_p = scalar_rmse(previous[:, :, 2], target[:, :, 2])

    gnn_u = scalar_rmse(predictions[:, :, 0], target[:, :, 0])
    gnn_v = scalar_rmse(predictions[:, :, 1], target[:, :, 1])
    gnn_p = scalar_rmse(predictions[:, :, 2], target[:, :, 2])

    # Velocity-vector RMSE
    baseline_velocity = np.sqrt(
        np.mean(
            (previous[:, :, 0] - target[:, :, 0]) ** 2
            + (previous[:, :, 1] - target[:, :, 1]) ** 2,
            axis=1,
        )
    )

    gnn_velocity = np.sqrt(
        np.mean(
            (predictions[:, :, 0] - target[:, :, 0]) ** 2
            + (predictions[:, :, 1] - target[:, :, 1]) ** 2,
            axis=1,
        )
    )

    gains = {
        "u": baseline_u / np.maximum(gnn_u, eps),
        "v": baseline_v / np.maximum(gnn_v, eps),
        "velocity": (
            baseline_velocity / np.maximum(gnn_velocity, eps)
        ),
        "p": baseline_p / np.maximum(gnn_p, eps),
    }

    baselines = {
        "u": baseline_u,
        "v": baseline_v,
        "velocity": baseline_velocity,
        "p": baseline_p,
    }

    gnn_errors = {
        "u": gnn_u,
        "v": gnn_v,
        "velocity": gnn_velocity,
        "p": gnn_p,
    }

    def print_block(title, mask):

        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)

        for name in ("u", "v", "velocity", "p"):

            b = baselines[name][mask]
            g = gnn_errors[name][mask]
            gain = gains[name][mask]

            if len(gain) == 0:
                continue

            better = gain > 1.0
            worse = gain < 1.0

            print(f"\n{name}")
            print(
                f"  Zero-delta RMSE median : {np.median(b):.6e}"
            )
            print(
                f"  GNN RMSE median        : {np.median(g):.6e}"
            )
            print(
                f"  Median gain            : {np.median(gain):.6e}"
            )
            print(
                f"  Mean gain              : {np.mean(gain):.6e}"
            )
            print(
                f"  Min gain               : {np.min(gain):.6e}"
            )
            print(
                f"  Max gain               : {np.max(gain):.6e}"
            )
            print(
                f"  GNN better             : "
                f"{better.sum()}/{len(gain)} "
                f"({100 * better.mean():.1f}%)"
            )
            print(
                f"  Zero-delta better      : "
                f"{worse.sum()}/{len(gain)} "
                f"({100 * worse.mean():.1f}%)"
            )

    all_mask = np.ones(len(time), dtype=bool)

    print_block(
        "ZERO-DELTA BASELINE vs GNN - ALL EVALUATED SAMPLES",
        all_mask,
    )

    # Early transient vs the rest
    early_mask = time < 0.5
    late_mask = time >= 0.5

    if early_mask.any():
        print_block(
            "ZERO-DELTA vs GNN - EARLY TRANSIENT (t < 0.5 s)",
            early_mask,
        )

    if late_mask.any():
        print_block(
            "ZERO-DELTA vs GNN - t >= 0.5 s",
            late_mask,
        )

    if held_out_mask is not None and not held_out_mask.all():
        print_block(
            "ZERO-DELTA vs GNN - HELD-OUT SAMPLES ONLY",
            held_out_mask,
        )

    return gains, baselines, gnn_errors