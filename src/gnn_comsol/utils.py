"""
General utilities used by experiment scripts.
"""

import argparse
import random

import numpy as np
import torch


def set_seeds(seed):
    """Seed Python, NumPy and PyTorch for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Some PyG scatter kernels have no deterministic CUDA version.
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as error:
        print(
            f"Note: deterministic algorithms unavailable ({error}). "
            "Runs stay seeded but are not bit-for-bit reproducible."
        )


def parse_args():
    """Parse command-line arguments for an experiment run."""

    parser = argparse.ArgumentParser(
        description="Run one GNN-COMSOL experiment."
    )

    parser.add_argument(
        "config",
        help="path to a YAML experiment file",
    )

    parser.add_argument(
        "--output-root",
        default="outputs",
        help="where to create the run directory (default: outputs)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not print a line per epoch",
    )

    return parser.parse_args()