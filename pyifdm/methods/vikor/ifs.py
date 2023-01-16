# Copyright (c) 2022 Jakub Więckowski

import numpy as np

def ifs(matrix, weights, types, normalization, distance, v):
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

            v : float
                Weights for the strategy of maximum group utility

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

    # postive and negative ideal solution
    pis, nis = np.zeros((nmatrix.shape[1], ), dtype=object), np.zeros((nmatrix.shape[1], ), dtype=object)
    for j in range(matrix.shape[1]):
        pis[j] = nmatrix[np.argmax(nmatrix[:, j, 0]), j]
        nis[j] = nmatrix[np.argmin(nmatrix[:, j, 0]), j]

    f = 1
    if 'normalized' in distance.__name__:
        f = 1/(2*matrix.shape[1])

    # validate data
    if any([all(p == n) for p, n in zip(pis, nis)]):
        raise ValueError('Matrix should not contain same values within a single column')

    # calculation of S and R rankings
    S, R, Q = np.zeros((nmatrix.shape[0], )), np.zeros((nmatrix.shape[0], )), np.zeros((nmatrix.shape[0], ))
    for i in range(matrix.shape[0]):
        if distance.__name__== 'normalized_euclidean_distance':
            S[i] = np.sum([weights[j] * (np.sqrt(f * distance(pis[j], nmatrix[i, j])) / np.sqrt(f * distance(pis[j], nis[j]))) for j in range(nmatrix.shape[1])])
            R[i] = np.max([weights[j] * (np.sqrt(f * distance(pis[j], nmatrix[i, j])) / np.sqrt(f * distance(pis[j], nis[j]))) for j in range(nmatrix.shape[1])])
        elif distance.__name__== 'normalized_hamming_distance':
            S[i] = np.sum([weights[j] * ((f * distance(pis[j], nmatrix[i, j])) / (f * distance(pis[j], nis[j]))) for j in range(nmatrix.shape[1])])
            R[i] = np.max([weights[j] * ((f * distance(pis[j], nmatrix[i, j])) / (f * distance(pis[j], nis[j]))) for j in range(nmatrix.shape[1])])
        else:
            S[i] = np.sum([weights[j] * (distance(pis[j], nmatrix[i, j]) / distance(pis[j], nis[j]))for j in range(nmatrix.shape[1])])
            R[i] = np.max([weights[j] * (distance(pis[j], nmatrix[i, j]) / distance(pis[j], nis[j]))for j in range(nmatrix.shape[1])])

    # calculation of the compromise ranking Q
    Q = v * ((S - np.min(S)) / (np.max(S) - np.min(S))) + (1 - v) * ((R - np.min(R)) / (np.max(R) - np.min(R)))

    return np.nan_to_num(S), np.nan_to_num(R), np.nan_to_num(Q)
