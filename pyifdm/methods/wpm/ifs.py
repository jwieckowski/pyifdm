# Copyright (c) 2023 Jakub Więckowski

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

    # weighted decision matrix
    wmatrix = matrix.copy()
    wmatrix[:, :, 0] = nmatrix[:, :, 0] ** weights
    wmatrix[:, :, 1] = 1 - (1 - nmatrix[:, :, 1])** weights

    # product
    Q  = np.prod(wmatrix, axis=0)
    
    # assessment score
    return np.array([score(q) for q in Q])
