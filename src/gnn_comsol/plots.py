"""
Figures.

Every function saves to a file and only optionally opens a window:
plt.show() is blocking, which stalls an unattended run.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def plot_loss(
    train_history,
    val_history,
    path,
    title,
):
    """
    Plot the training and validation loss.

    Two figures are always written:

        <name>.png
        <name>_log.png

    The first uses a linear y-axis, the second a logarithmic y-axis.
    """

    import matplotlib.pyplot as plt
    from pathlib import Path

    path = Path(path)

    epochs_train = range(1, len(train_history) + 1)

    # ========================================================
    # LINEAR SCALE
    # ========================================================

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        epochs_train,
        train_history,
        label="Training loss",
    )

    if val_history:
        epochs_val = range(1, len(val_history) + 1)

        ax.plot(
            epochs_val,
            val_history,
            label="Validation loss",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # LOGARITHMIC SCALE
    # ========================================================

    log_path = path.with_name(
        f"{path.stem}_log{path.suffix}"
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        epochs_train,
        train_history,
        label="Training loss",
    )

    if val_history:
        epochs_val = range(1, len(val_history) + 1)

        ax.plot(
            epochs_val,
            val_history,
            label="Validation loss",
        )

    ax.set_yscale("log")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title} — log scale")

    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()
    fig.savefig(
        log_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)