"""Data-loss functions used during training."""

import math

import torch


def mse_topk_loss(
    prediction,
    target,
    topk_fraction=0.10,
    topk_weight=1.0,
    vector_error=False,
):
    """
    Standard MSE plus the mean squared error over the worst fraction
    of nodes.

    For vector quantities (u, v), nodes are ranked using their mean
    squared component error, so u and v belonging to the same node
    remain together.
    """

    if not 0.0 < topk_fraction <= 1.0:
        raise ValueError(
            "topk_fraction must satisfy 0 < topk_fraction <= 1."
        )

    if topk_weight < 0.0:
        raise ValueError(
            "topk_weight must be non-negative."
        )

    squared_error = (prediction - target).square()

    mse_loss = squared_error.mean()

    if vector_error:
        # [B, N, 2] -> [B, N]
        ranking_error = squared_error.mean(dim=-1)
    else:
        # [B, N, 1] -> [B, N]
        ranking_error = squared_error.squeeze(-1)

    num_nodes = ranking_error.shape[1]

    k = max(
        1,
        math.ceil(topk_fraction * num_nodes),
    )

    topk_per_sample = torch.topk(
        ranking_error,
        k=k,
        dim=1,
        largest=True,
        sorted=False,
    ).values.mean(dim=1)

    topk_loss = topk_per_sample.mean()

    total_loss = mse_loss + topk_weight * topk_loss

    return total_loss, mse_loss, topk_loss