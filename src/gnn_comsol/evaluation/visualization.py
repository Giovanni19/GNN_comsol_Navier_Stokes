"""
Visualization utilities for model evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import animation
from matplotlib.colors import LogNorm

def contiguous_runs(mask):
    """The [start, end) index ranges where `mask` is True."""

    runs = []
    start = None

    for i, flag in enumerate(mask):

        if flag and start is None:
            start = i

        elif not flag and start is not None:
            runs.append((start, i))
            start = None

    if start is not None:
        runs.append((start, len(mask)))

    return runs

def plot_rmse(rmse, time, output_dir, held_out_mask=None):
    """
    RMSE over time, with the held-out region shaded.
    """

    everything_held_out = (
        held_out_mask is None or bool(held_out_mask.all())
    )

    prefix = "test_rmse" if everything_held_out else "trajectory_rmse"

    for name, values in rmse.items():

        for scale in ("linear", "log"):

            figure, axes = plt.subplots(figsize=(10, 5))

            axes.plot(time, values)

            if not everything_held_out:

                for first, last in contiguous_runs(held_out_mask):

                    axes.axvspan(
                        time[first],
                        time[min(last, len(time) - 1)],
                        alpha=0.15,
                        color="tab:green",
                        label="_nolegend_",
                    )

            if scale == "log":
                axes.set_yscale("log")

            axes.set_xlabel("Time")
            axes.set_ylabel(f"RMSE {name}")

            axes.set_title(
                (
                    f"Held-out set - {name} RMSE over time"
                    if everything_held_out
                    else f"Whole trajectory - {name} RMSE over time "
                         "(shaded: held out; the rest was trained on)"
                )
                + (" (log scale)" if scale == "log" else "")
            )

            axes.grid(True)

            figure.tight_layout()

            suffix = "_log" if scale == "log" else ""

            figure.savefig(
                output_dir / f"{prefix}_{name}{suffix}.png",
                dpi=200
            )

def mesh_triangulation(evaluation):
    """
    A triangulation that follows the mesh instead of the convex hull.

    A Delaunay triangle is kept only if its three edges are in the
    graph, so no triangle spans a cavity or an obstacle.
    """

    triangulation = mtri.Triangulation(
        evaluation.pos[:, 0],
        evaluation.pos[:, 1]
    )

    edge_index = np.asarray(evaluation.edge_index)

    if edge_index.shape[0] != 2:
        edge_index = edge_index.T

    edges = {
        (int(min(i, j)), int(max(i, j)))
        for i, j in edge_index.T
    }

    def has_edge(i, j):
        return (min(i, j), max(i, j)) in edges

    triangles = triangulation.triangles

    mask = np.array(
        [
            not (
                has_edge(i, j)
                and has_edge(j, k)
                and has_edge(k, i)
            )
            for i, j, k in triangles
        ]
    )

    triangulation.set_mask(mask)

    print(
        f"Visualization triangulation: {len(triangles)} triangles | "
        f"{int(mask.sum())} removed | {int((~mask).sum())} retained"
    )

    return triangulation


def _animate_two_error_fields(
    triangulation,
    fields,
    titles,
    colorbar_label,
    time,
    output_path,
    log_scale=False,
):
    """
    Animate two nodal error fields side by side using one common scale.

    The two panels are:
        1. GNN prediction vs. ground truth
        2. Previous time step vs. ground truth

    The colorbar is placed in a dedicated axis so that it does not
    overlap with either mesh plot.
    """

    combined = np.concatenate([field.ravel() for field in fields])

    # ------------------------------------------------------------
    # Colour normalization
    # ------------------------------------------------------------

    if log_scale:

        positive = combined[combined > 0.0]

        if positive.size == 0:
            print(
                "WARNING: all error values are zero; "
                "cannot create logarithmic animation."
            )
            return

        vmin = float(np.percentile(positive, 1))
        vmax = float(np.percentile(positive, 99))

        if vmin <= 0.0:
            vmin = float(positive.min())

        if vmax <= vmin:
            vmax = float(positive.max())

        norm = LogNorm(vmin=vmin, vmax=vmax)

        print(
            f"Two-panel LOG color scale: "
            f"{vmin:.3e} -> {vmax:.3e}"
        )

    else:

        vmin = 0.0
        vmax = float(np.percentile(combined, 99))

        if vmax <= 0.0:
            vmax = 1.0

        norm = None

        print(
            f"Two-panel LINEAR color scale: "
            f"{vmin:.3e} -> {vmax:.3e}"
        )

    # ------------------------------------------------------------
    # Figure
    #
    # Three columns:
    #   mesh 1 | mesh 2 | colorbar
    #
    # The third column is deliberately narrow and reserved entirely
    # for the colorbar.
    # ------------------------------------------------------------

    figure = plt.figure(figsize=(15, 5.5))

    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=[1, 1, 0.045],
        wspace=0.20,
    )

    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
    ]

    colorbar_axis = figure.add_subplot(grid[0, 2])

    # ------------------------------------------------------------
    # Initial plot
    # ------------------------------------------------------------

    if log_scale:

        first = axes[0].tripcolor(
            triangulation,
            fields[0][0],
            shading="flat",
            norm=norm,
            cmap="viridis",
        )

    else:

        first = axes[0].tripcolor(
            triangulation,
            fields[0][0],
            shading="flat",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )

    colorbar = figure.colorbar(
        first,
        cax=colorbar_axis,
    )

    if log_scale:
        colorbar.set_label(
            colorbar_label + " (log scale)"
        )
    else:
        colorbar.set_label(colorbar_label)

    # ------------------------------------------------------------
    # Animation update
    # ------------------------------------------------------------

    def update(frame):

        artists = []

        for axis, field, title in zip(
            axes,
            fields,
            titles,
        ):

            axis.clear()

            if log_scale:

                plot = axis.tripcolor(
                    triangulation,
                    field[frame],
                    shading="flat",
                    norm=norm,
                    cmap="viridis",
                )

            else:

                plot = axis.tripcolor(
                    triangulation,
                    field[frame],
                    shading="flat",
                    vmin=vmin,
                    vmax=vmax,
                    cmap="viridis",
                )

            rms = np.sqrt(
                np.mean(field[frame] ** 2)
            )

            axis.set_aspect(
                "equal",
                adjustable="box",
            )

            axis.set_xlabel("x")
            axis.set_ylabel("y")

            axis.set_title(
                f"{title}\n"
                f"RMS difference = {rms:.3e}"
            )

            artists.append(plot)

        scale_name = (
            "logarithmic colour scale"
            if log_scale
            else "linear colour scale"
        )

        figure.suptitle(
            f"Initial-guess comparison at target time "
            f"t = {time[frame]:.3f} s\n"
            f"{scale_name}",
            fontsize=14,
        )

        return tuple(artists)

    figure_animation = animation.FuncAnimation(
        figure,
        update,
        frames=range(len(time)),
        interval=50,
        blit=False,
    )

    # Leave enough room for the title without moving the colorbar
    # over the mesh axes.
    figure.subplots_adjust(
        top=0.78,
        bottom=0.15,
        left=0.06,
        right=0.95,
    )

    figure_animation.save(
        output_path,
        writer=animation.PillowWriter(fps=15),
    )

    plt.close(figure)

    print(
        f"Two-panel error animation saved to: "
        f"{output_path}"
    )

def animate_pressure_error(
    evaluation,
    prediction_target_error,
    previous_target_error,
    time,
    output_dir,
):
    triangulation = mesh_triangulation(evaluation)

    fields = (
        prediction_target_error,
        previous_target_error,
    )

    titles = (
        "GNN prediction vs. ground truth",
        "Previous time step vs. ground truth",
    )

    # Linear scale
    _animate_two_error_fields(
        triangulation,
        fields=fields,
        titles=titles,
        colorbar_label="Absolute pressure difference",
        time=time,
        output_path=(
            output_dir /
            "pressure_error_comparison_animation.gif"
        ),
        log_scale=False,
    )

    # Logarithmic scale
    _animate_two_error_fields(
        triangulation,
        fields=fields,
        titles=titles,
        colorbar_label="Absolute pressure difference",
        time=time,
        output_path=(
            output_dir /
            "pressure_error_comparison_animation_log.gif"
        ),
        log_scale=True,
    )

def animate_velocity_error(
    evaluation,
    prediction_target_error,
    previous_target_error,
    time,
    output_dir,
):
    triangulation = mesh_triangulation(evaluation)

    fields = (
        prediction_target_error,
        previous_target_error,
    )

    titles = (
        "GNN prediction vs. ground truth",
        "Previous time step vs. ground truth",
    )

    # Linear scale
    _animate_two_error_fields(
        triangulation,
        fields=fields,
        titles=titles,
        colorbar_label="Velocity-vector difference magnitude",
        time=time,
        output_path=(
            output_dir /
            "velocity_error_comparison_animation.gif"
        ),
        log_scale=False,
    )

    # Logarithmic scale
    _animate_two_error_fields(
        triangulation,
        fields=fields,
        titles=titles,
        colorbar_label="Velocity-vector difference magnitude",
        time=time,
        output_path=(
            output_dir /
            "velocity_error_comparison_animation_log.gif"
        ),
        log_scale=True,
    )
