"""
What the evaluation reports, and what it calls it.

evaluate_test.py evaluates every transition of a simulation, including
the ones the run trained on: the whole trajectory is worth looking at,
and the initial pressure transient is the part most worth looking at.

That makes the headline numbers a diagnostic and not a measure of
generalisation, and for a while nothing said so - the figures were
saved as "test_rmse", and the run printed that every sample had been
held out. These tests exist so that the two can never again be
reported as the same thing.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from gnn_comsol.evaluation.reporting import report
from gnn_comsol.evaluation.setup import held_out_samples
from gnn_comsol.evaluation.visualization import contiguous_runs


@pytest.fixture(scope="module")
def evaluate_test():
    """The functions evaluate_test.py uses, from gnn_comsol.evaluation."""

    return SimpleNamespace(
        report=report,
        held_out_samples=held_out_samples,
        contiguous_runs=contiguous_runs,
    )


@pytest.fixture
def rmse():
    # u grows along the trajectory: the held-out tail is the worst part,
    # so averaging over everything flatters the model.
    return {
        "u": np.linspace(1.0, 2.0, 20),
        "v": np.ones(20),
        "p": np.ones(20),
    }


def test_a_whole_simulation_split_reports_one_set_of_numbers(
    evaluate_test, rmse, capsys
):
    """
    Everything was held out, so there is no distinction to draw and
    nothing to warn about.
    """

    evaluate_test.report(rmse, np.linspace(0, 1, 20), None)

    out = capsys.readouterr().out

    assert "HELD-OUT RMSE" in out
    assert "WERE trained on" not in out


def test_a_partial_split_reports_the_two_apart(
    evaluate_test, rmse, capsys
):
    """
    The diagnostic figure and the generalisation figure are different
    numbers, and the second must be present and labelled.
    """

    mask = np.zeros(20, dtype=bool)
    mask[15:] = True

    evaluate_test.report(rmse, np.linspace(0, 1, 20), mask)

    out = capsys.readouterr().out

    assert "WHOLE TRAJECTORY" in out
    assert "WERE trained on" in out
    assert "HELD-OUT RMSE - PHYSICAL UNITS (5 of 20 samples)" in out

    # The held-out mean must be the tail's mean, not the whole curve's
    whole = rmse["u"].mean()
    held_out = rmse["u"][mask].mean()

    assert held_out > whole

    assert f"{held_out:.6e}" in out
    assert f"{whole:.6e}" in out


def test_an_empty_held_out_set_says_so(evaluate_test, rmse, capsys):

    evaluate_test.report(
        rmse,
        np.linspace(0, 1, 20),
        np.zeros(20, dtype=bool),
    )

    out = capsys.readouterr().out

    assert "No sample was held out" in out


@pytest.mark.parametrize(
    "mask, expected",
    [
        ([0, 0, 1, 1, 0], [(2, 4)]),
        ([1, 1, 0, 0, 1], [(0, 2), (4, 5)]),
        ([1, 1, 1], [(0, 3)]),
        ([0, 0], []),
        ([1, 0, 1, 0, 1], [(0, 1), (2, 3), (4, 5)]),
    ],
)
def test_contiguous_runs_finds_the_shaded_regions(
    evaluate_test, mask, expected
):
    """
    The figure shades the held-out region. A random split scatters it,
    so the runs must be found rather than assumed to be one block.
    """

    assert evaluate_test.contiguous_runs(
        np.array(mask, dtype=bool)
    ) == expected


# ---------------------------------------------------------------------
# Which samples were held out, when inference skips differently
# ---------------------------------------------------------------------

NUM_TRANSITIONS = 200


def loaded_with(skip_initial):
    """A RawDataset with the samples left after skip_initial."""

    from gnn_comsol.data.loading import RawDataset

    num_samples = NUM_TRANSITIONS - skip_initial

    return RawDataset(
        X_input=np.zeros((num_samples, 2, 3)),
        Y_target=np.zeros((num_samples, 2, 3)),
        edge_index=np.zeros((2, 1), dtype=int),
        edge_weight=np.ones(1),
        delta_t=np.ones(num_samples),
        pos=np.zeros((2, 2)),
    )


def run_config(mode, skip_initial):
    return {
        "seed": 68,
        "dataset": {"skip_initial": skip_initial},
        "split": {
            "mode": mode,
            "train_fraction": 0.70,
            "val_fraction": 0.15,
            "gap": 1,
        },
    }


def true_test_transitions(mode, training_skip):
    """The transitions j (X[j] -> X[j+1]) the training run held out."""

    from gnn_comsol.data.splitting import compute_split_indices

    _, _, test = compute_split_indices(
        NUM_TRANSITIONS - training_skip, mode=mode, seed=68
    )

    return test + training_skip


@pytest.mark.parametrize("mode", ["temporal", "random"])
@pytest.mark.parametrize("inference_skip", [0, 10, 49])
def test_held_out_samples_follow_the_training_split(
    evaluate_test, mode, inference_skip
):
    """
    Trained with skip_initial=49, evaluated with another skip: the
    held-out samples must be the ones the run held out, not a split
    recomputed on a different number of samples.
    """

    training_skip = 49

    _, indices = evaluate_test.held_out_samples(
        loaded_with(inference_skip),
        run_config(mode, training_skip),
        mode,
        skip_initial=inference_skip,
    )

    assert sorted(indices + inference_skip) == sorted(
        true_test_transitions(mode, training_skip)
    )


def test_transitions_skipped_in_training_are_not_held_out(evaluate_test):
    """Evaluating from a later snapshot drops the test samples before it."""

    _, indices = evaluate_test.held_out_samples(
        loaded_with(190),
        run_config("temporal", 49),
        "temporal",
        skip_initial=190,
    )

    expected = true_test_transitions("temporal", 49)

    assert sorted(indices + 190) == sorted(expected[expected >= 190])
    assert np.all(indices >= 0)
