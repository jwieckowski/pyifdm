# Copyright (c) 2022 Jakub Więckowski

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

            normalization : callable
                Function used to calculate normalized decision matrix

            score : callable
                Function used to calculate crisp score of IFS 
        Returns
        -------
            ndarray
                Crisp preferences of alternatives

    """
    # optimal preference ranking
    R = np.zeros((matrix.shape[1], matrix.shape[2]), dtype=object)
    for j in range(matrix.shape[1]):
        if types[j] == 1:
            R[j] = matrix[np.argmax(matrix[:, j, 0]), j]
        else:
            R[j] = matrix[np.argmin(matrix[:, j, 0]), j]

    # extended decision matrix
    exmatrix = np.ones((matrix.shape[0]+1, matrix.shape[1], matrix.shape[2]), dtype=object)
    exmatrix[0] = R
    exmatrix[1:] = matrix

    # normalized matrix
    nmatrix = normalization(exmatrix, types)

    # weighted normalized matrix
    wmatrix = np.zeros((nmatrix.shape[0], nmatrix.shape[1], nmatrix.shape[2]), dtype=object)

    # crisp weights
    if weights.ndim == 1:
        weights = np.repeat(weights, 2).reshape((len(weights), 2))

    wmatrix[:, :, 0] = 1 - (1 - nmatrix[:, :, 0])**weights[:, 0]
    wmatrix[:, :, 1] = nmatrix[:, :, 1]**weights[:, 1]

    # score values
    S = score(wmatrix)

    # overal performance rating
    M = np.sum(S, axis=1)

    Q = M[1:] / M[0]
    return Q
