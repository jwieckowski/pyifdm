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

    # crisp weights
    if weights.ndim == 1:
        weights = np.repeat(weights, 2).reshape((len(weights), 2))

    # WSM-based calculations
    wsm_p = 1 - np.prod((1 - nmatrix[:, :, 0]) ** weights[:, 0], axis=1)
    wsm_q = np.prod((nmatrix[:, :, 1] ** weights[:, 1]), axis=1)
    Q1 = 1/2 * (score(np.vstack([wsm_p, wsm_q]).T) + 1)

    # WPM-based calculations
    wpm_p = np.prod(nmatrix[:, :, 0] ** weights[:, 0], axis=1)
    wpm_q = 1 - np.prod((1 - nmatrix[:, :, 1]) ** weights[:, 1], axis=1)
    Q2 = 1/2 * (score(np.vstack([wpm_p, wpm_q]).T) + 1)

    # assessment score
    return np.array(v * Q1 + (1 - v) * Q2)
