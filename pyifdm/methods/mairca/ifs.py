# Copyright (c) 2022 Jakub Więckowski

import numpy as np

def ifs(matrix, weights, types, normalization, distance, score):
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

            score: callable
                Function used to calculate crisp score of IFS
                
        Returns
        -------
            ndarray
                Crisp preferences of alternatives

    """

    f = 1
    if 'normalized' in distance.__name__:
        f = 1/(2*matrix.shape[1])

    # distance measures
    dm = np.zeros((matrix.shape[0], matrix.shape[1], 2), dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if distance.__name__ == 'normalized_euclidean_distance':
                dm[i, j, 0] = np.sqrt(f * distance(matrix[i, j], np.array([1, 0, 0])))
                dm[i, j, 1] = np.sqrt(f * distance(matrix[i, j], np.array([0, 1, 0])))
            elif distance.__name__ == 'normalized_hamming_distance':
                dm[i, j, 0] = f * distance(matrix[i, j], np.array([1, 0, 0]))
                dm[i, j, 1] = f * distance(matrix[i, j], np.array([0, 1, 0]))
            else:
                dm[i, j, 0] = distance(matrix[i, j], np.array([1, 0, 0]))
                dm[i, j, 1] = distance(matrix[i, j], np.array([0, 1, 0]))

    # normalization condition for different methods than in reference research paper
    if normalization.__name__ != 'minmax_normalization':
        dm =  normalization(dm, types)

    # closeness coefficient
    cw = dm[:, :, 1] / (dm[:, :, 1] + dm[:, :, 0])

    # normalized matrix
    if normalization.__name__ == 'minmax_normalization':
        nmatrix = normalization(cw, types)
    else:
        nmatrix = cw

    # crisp weights
    if weights.ndim == 2:
        weights = np.array([score(w) for w in weights])

    # theoretical intuitionistic fuzzy decision matrix
    tdm = 1 / matrix.shape[0] * nmatrix * weights
    tdm = np.tile(np.max(tdm, axis=0), nmatrix.shape[0]).reshape((nmatrix.shape[0], nmatrix.shape[1]))

    # real evaluation matrix
    rem = nmatrix * tdm

    # gap matrix
    gm = tdm - rem

    # utility score
    s = np.sum(gm, axis=1)
    return s
