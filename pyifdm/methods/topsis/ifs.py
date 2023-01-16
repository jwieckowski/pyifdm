# Copyright (c) 2022 Jakub Więckowski

import numpy as np

def ifs(matrix, weights, types, normalization, distance):
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

    # weighted matrix
    wmatrix = np.zeros((nmatrix.shape[0], nmatrix.shape[1], 3), dtype=object)

    # crisp weights
    if weights.ndim == 1:
        weights = np.repeat(weights, 2).reshape((len(weights), 2))

    wmatrix[:, :, 0] = nmatrix[:, :, 0] * weights[:, 0]
    wmatrix[:, :, 1] = nmatrix[:, :, 1] + weights[:, 1] - nmatrix[:, :, 1] * weights[:, 1]
    wmatrix[:, :, 2] = 1 - nmatrix[:, :, 1] - weights[:, 1] - nmatrix[:, :, 0] * weights[:, 0] + nmatrix[:, :, 1]  * weights[:, 1]

    # closeness to intuitionistic fuzzy positive and negative ideal solution
    aplus, aminus = np.zeros((matrix.shape[1], ), dtype=object), np.zeros((matrix.shape[1], ), dtype=object)
    for j in range(matrix.shape[1]):
        if types[j] == 1:
            aplus[j] = wmatrix[np.argmax(wmatrix[:, j, 0]), j]
            aminus[j] = wmatrix[np.argmin(wmatrix[:, j, 0]), j]
        else:
            aplus[j] = wmatrix[np.argmin(wmatrix[:, j, 0]), j]
            aminus[j] = wmatrix[np.argmax(wmatrix[:, j, 0]), j]

    f = 1
    if 'normalized' in distance.__name__:
        f = 1/(2*matrix.shape[1])

    # distance from ideal solution
    splus, sminus = np.zeros((wmatrix.shape[0], )), np.zeros((wmatrix.shape[0], ))
    for i in range(wmatrix.shape[0]):
        if distance.__name__== 'normalized_euclidean_distance':
            splus[i] = np.sqrt(f * np.sum([distance(wmatrix[i, idx], ap) for idx, ap in enumerate(aplus)]))
            sminus[i] = np.sqrt(f * np.sum([distance(wmatrix[i, idx], am) for idx, am in enumerate(aminus)]))
        elif distance.__name__== 'normalized_hamming_distance':
            splus[i] = f * np.sum([distance(wmatrix[i, idx], ap) for idx, ap in enumerate(aplus)])
            sminus[i] = f * np.sum([distance(wmatrix[i, idx], am) for idx, am in enumerate(aminus)])
        else:
            splus[i] = np.sum([distance(wmatrix[i, idx], ap) for idx, ap in enumerate(aplus)])
            sminus[i] = np.sum([distance(wmatrix[i, idx], am) for idx, am in enumerate(aminus)])

    # assessment score
    return sminus / (splus + sminus)
