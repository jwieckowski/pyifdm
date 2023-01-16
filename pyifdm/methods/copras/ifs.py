# Copyright (c) 2022 Jakub Więckowski
# Copyright (c) 2022 Bartłomiej Kizielewicz

import numpy as np

def ifs(matrix, weights, types, normalization, score):
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
                Function used to calculate normalized decision matrix

            score: callable
                Function used to calculate crisp score of IFS

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

    # weighted matrix
    wmatrix = np.zeros(nmatrix.shape)

    # crisp weights
    if weights.ndim == 1:
        weights = np.repeat(weights, 2).reshape((len(weights), 2))

    wmatrix[:, :, 0] = np.sqrt(1 - (1 - nmatrix[:, :, 0] ** 2) ** weights[:, 0])
    wmatrix[:, :, 1] = np.sqrt((nmatrix[:, :, 1] ** 2) ** weights[:, 1])

    # Score function
    s = score(wmatrix)

    # Determine the maximizing and minimizing index
    Sp = np.sum(s[:, types == 1], axis=1) / types[types == 1].shape
    Sr = np.sum(s[:, types == -1], axis=1) / types[types == -1].shape

    # Determine the relative significance value of each alternative
    N = np.sum(np.exp(Sr)) / np.sum(1 / np.exp(Sr))
    Q =  Sp + (N / np.exp(Sr))

    return Q / np.max(Q)
