# Copyright (c) 2023 Bartłomiej Kizielewicz

import numpy as np


def ifs(matrix, weights, types, normalization, score, v):
    """
        Calculates the alternatives preferences based on Intuitionistic Fuzzy Sets

        Parameters
        ----------
            matrix : ndarray
                Decision matrix / alternatives data.
                Alternatives are in rows and Criteria are in columns.

            weights : ndarray
                Vector of criteria weights in a crisp or Intuitionistic Fuzzy form

            types : ndarray
                Types of criteria, 1 profit, -1 cost

            normalization: callable
                Function used to normalize the decision matrix

            distance: callable
                Function used to calculate distance between two IFS

        Returns
        -------
            ndarray
                Crisp preferences of alternatives

    """

    # normalized matrix
    if normalization is not None:
        nmatrix = normalization(matrix, types)
    else:
        nmatrix = matrix.copy()

    print(nmatrix)
    wsm_p = 1 - np.prod((1 - nmatrix[:, :, 0]) ** weights, axis=1)
    wsm_q = np.prod((nmatrix[:, :, 1] ** weights), axis=1)
    Q1 = 1/2 * (score(np.vstack([wsm_p, wsm_q]).T) + 1)
    print(Q1)

    wpm_p = np.prod(nmatrix[:, :, 0] ** weights, axis=1)
    wpm_q = 1 - np.prod((1 - nmatrix[:, :, 1]) ** weights, axis=1)
    Q2 = 1/2 * (score(np.vstack([wpm_p, wpm_q]).T) + 1)

    # assessment score
    return np.array(v * Q1 + (1 - v) * Q2)



def minmax_normalization(matrix, types):
    """
        Calculates the normalized value of Intuitionistic Fuzzy matrix using Min-Max normalization

        Parameters
        ----------
            matrix : ndarray
                Matrix with Intuitionistic Fuzzy Sets

            types : ndarray
                Types of criteria, 1 profit, -1 cost

        Returns
        -------
            ndarray
                Normalized Intuitionistic Fuzzy matrix
    """

    cmax = np.max(matrix, axis=0)
    cmin = np.min(matrix, axis=0)

    # validate data
    if np.min(cmax[types == 1] - cmin[types == 1]) == 0 or np.min(cmax[types == -1] - cmin[types == -1]) == 0:
        raise ValueError('Subtraction result of matrix elements cannot equal 0')

    nmatrix = np.zeros((matrix.shape))
    nmatrix[:, types == 1] = (matrix[:, types == 1] - cmin[types == 1]) / (cmax[types == 1] - cmin[types == 1])
    nmatrix[:, types == -1] = (cmax[types == -1] - matrix[:, types == -1]) / (cmax[types == -1] - cmin[types == -1])

    return nmatrix.astype(float)


def chen_score_1(a):
    """
        Calculates score of the Intuitionistic Fuzzy Set (u, v) and returns a crisp value.
        Uses a formula: (u - v)

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Set (u, v)

        Returns
        -------
            float
                Crisp value
    """
    # cast types
    a = a.astype(float)

    if a.ndim == 1:
        return a[0] - a[1]
    elif a.ndim == 2:
        return a[:, 0] - a[:, 1]
    else:
        return a[:, :, 0] - a[:, :, 1]

data = [
    [(0.6268, 0.2794), (0.5987, 0.3323), (0.7585, 0.1520), (0.2709, 0.6292), (0.4841, 0.4452), (0.1392, 0.8182), (0.6958, 0.2094), (0.5743, 0.3682)],
    [(0.4520, 0.4711), (0.6728, 0.2196), (0.7207, 0.1866), (0.1020, 0.8858), (0.1500, 0.8017), (0.3508, 0.5584), (0.5935, 0.3042), (0.5346, 0.4087)],
    [(0.1050, 0.8538), (0.0774, 0.9226), (0.6874, 0.2157), (0.1970, 0.7652), (0.4596, 0.4505), (0.5343, 0.4121), (0.3393, 0.5668), (0.1498, 0.7599)],
    [(0.2575, 0.6433), (0.1673, 0.7376), (0.7207, 0.1866), (0.4160, 0.5038), (0.4841, 0.4452), (0.1190, 0.8702), (0.8330, 0.1301), (0.1259, 0.8351)],
    [(0.2181, 0.6829), (0.0774, 0.9226), (0.6164, 0.2930), (0.5278, 0.4182), (0.2516, 0.6476), (0.0912, 0.8934), (0.8162, 0.1474), (0.1696, 0.7357)]
]
table = np.array(data)

data = [0.1471, 0.1618, 0.0882, 0.1280, 0.1105, 0.1132, 0.1132, 0.1379]

# Konwersja danych na tablicę numpy
weights = np.array(data)

types = np.array([-1, -1, 1, -1, -1, -1, 1, -1])

print(ifs(table, weights, types, lambda vec, types: vec, chen_score_1, 0.5))