# Copyright (c) 2022 Jakub Więckowski

import numpy as np

__all__ = [
    'rank',
    'generate_ifs_matrix'
]


def rank(x, descending=True):
    """
        Calculates ranking of given values with the given direction, default descending order

        Parameters
        ----------
            x: ndarray
                Array with values

            descending: boolean, default=True
                Switch to change ranking order

        Returns
        -------
            ndarray
                Ranking with given order

    """
    s = [sorted(x, reverse=descending).index(r)+1 for r in x]
    return np.array([(ss * s.count(ss) + s.count(ss) - 1) / s.count(ss) if s.count(ss) <= 2 else np.sum(list(range(ss, ss+s.count(ss)))) / s.count(ss) for ss in s])


def generate_ifs_matrix(m, n):
    """
        Generates random Intuitionistic Fuzzy matrix with m alternatives and n criteria

        Parameters
        ----------
            m: int
                Number of alternatives

            n: int
                Number of criteria

        Returns
        -------
            ndarray
                Matrix with random IFSs

    """

    arr = []
    while len(arr) < m*n:
        t = np.random.random((2, 1))
        if np.sum(t) <= 1:
            arr.append(t)
    return np.array(arr, dtype=object).reshape((m, n, 2)).astype(float)

