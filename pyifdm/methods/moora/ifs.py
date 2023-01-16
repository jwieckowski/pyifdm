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

            normalization: callable
                Function used to normalize the decision matrix

            score: callable, default
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
    wmatrix = np.zeros((nmatrix.shape[0], nmatrix.shape[1], 3), dtype=object)

    # crisp weights
    if weights.ndim == 1:
        weights = np.repeat(weights, 2).reshape((len(weights), 2))

    wmatrix[:, :, 0] = nmatrix[:, :, 0] * weights[:, 0]
    wmatrix[:, :, 1] = nmatrix[:, :, 1] + weights[:, 1] - nmatrix[:, :, 1] * weights[:, 1]
    wmatrix[:, :, 2] = 1 - wmatrix[:, :, 1] - wmatrix[:, :, 0]

    # sum of costs and benefits
    Sp, Sm  = np.zeros((matrix.shape[0], 3)), np.zeros((matrix.shape[0], 3))
    profit_indexes = np.where(types == 1)
    cost_indexes = np.where(types == -1)
    for i in range(wmatrix.shape[0]):
        profit = wmatrix[i, profit_indexes[0][0]]
        cost = wmatrix[i, cost_indexes[0][0]]

        for idx in profit_indexes[0][1:]:
            profit[0] = profit[0] + wmatrix[i, idx, 0] - profit[0] * wmatrix[i, idx, 0]
            profit[1] *= wmatrix[i, idx, 1]

        for idx in cost_indexes[0][1:]:
            cost[0] = cost[0] + wmatrix[i, idx, 0] - cost[0] * wmatrix[i, idx, 0]
            cost[1] *= wmatrix[i, idx, 1]

        # Sp.append([profit[0], profit[1], 1 - profit[0] - profit[1]])
        # Sm.append([cost[0], cost[1], 1 - cost[0] - cost[1]])
        Sp[i] = [profit[0], profit[1], 1 - profit[0] - profit[1]]
        Sm[i] = [cost[0], cost[1], 1 - cost[0] - cost[1]]

    # Sp, Sm = np.array(Sp), np.array(Sm) 

    # score functions
    Dp = score(Sp)
    Dm = score(Sm)

    return Dp - Dm
