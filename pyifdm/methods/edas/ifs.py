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
                Function used to calculate normalized decision matrix

            score: callable
                Function used to calculate crisp score of IFS

        Returns
        -------
            ndarray
                Crisp preferences of alternatives

    """
    
    # normalized matrix
    nmatrix = normalization(matrix, types)

    # average solution
    av = np.array([[1 - (np.prod(1 - nmatrix[:, j, 0]))**(1/nmatrix.shape[0]),
                  (np.prod(nmatrix[:, j, 1]))**(1/nmatrix.shape[0])] for j in range(nmatrix.shape[1])])

    # positive and negative distances from average
    pda, nda = np.zeros((nmatrix.shape[0], nmatrix.shape[1])), np.zeros((nmatrix.shape[0], nmatrix.shape[1]))
    for i in range(nmatrix.shape[0]):
        for j in range(nmatrix.shape[1]):
            pda[i, j] = np.max([0, (score(nmatrix[i, j]) - score(av[j]))]) / score(av[j])
            nda[i, j] = np.max([0, (score(av[j]) - score(nmatrix[i, j]))]) / score(av[j])

    # crisp weights
    if weights.ndim == 2:
        weights = np.array([score(w) for w in weights])

    # weighted positive and negative distances
    sp = np.sum(weights * pda, axis=1)
    sn = np.sum(weights * nda, axis=1)

    # normalized weighted positive and negative distances
    nsp = sp / np.max(sp) if np.max(sp) != 0 else np.zeros((sp.shape))
    nsn = 1 - sn / np.max(sn) if np.max(sn) != 0 else np.zeros((sn.shape))

    # appraisal score
    return 1/2 * (nsp + nsn)
