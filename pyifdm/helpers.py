# Copyright (c) 2022-2026 Jakub Więckowski & contributors

from __future__ import annotations

from typing import Union

import numpy as np

__all__ = [
    "rank",
    "generate_ifs_matrix",
]


def rank(
    x: Union[np.ndarray, list],
    descending: bool = True,
) -> np.ndarray:
    """
    Calculate ranking of values (average ranks for ties).

    Parameters
    ----------
    x :
        Preference values.
    descending :
        If True, higher values receive better (lower) ranks.

    Returns
    -------
    ndarray
        Ranking starting from 1.
    """
    x = np.asarray(x, dtype=float)
    # argsort twice gives ranks; handle ties by averaging
    order = np.argsort(-x if descending else x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)

    # average ties
    unique, inverse, counts = np.unique(
        x, return_inverse=True, return_counts=True
    )
    for i, c in enumerate(counts):
        if c > 1:
            idx = inverse == i
            ranks[idx] = ranks[idx].mean()
    return ranks


def generate_ifs_matrix(m: int, n: int) -> np.ndarray:
    """
    Generate a random IFS decision matrix of shape (m, n, 2).

    Membership and non-membership are drawn uniformly such that
    μ + v ≤ 1.
    """
    matrix = np.zeros((m, n, 2))
    for i in range(m):
        for j in range(n):
            while True:
                mu, nu = np.random.random(2)
                if mu + nu <= 1.0:
                    matrix[i, j] = [mu, nu]
                    break
    return matrix
