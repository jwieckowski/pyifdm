# Copyright (c) 2023 Jakub Więckowski

import numpy as np

def ifs(matrix, weights, types):
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
        Returns
        -------
            ndarray
                Crisp preferences of alternatives

    """

    # aggregated IF decision matrix
    matrix_p =  np.sqrt((matrix[:, :, 0] - 1)**2 + (matrix[:, :, 1] - 0)**2 + (matrix[:, :, 2] - 0)**2)
    matrix_m =  np.sqrt((matrix[:, :, 0] - 0)**2 + (matrix[:, :, 1] - 1)**2 + (matrix[:, :, 2] - 0)**2)
    if_matrix = matrix_m / (matrix_m + matrix_p)

    # Extended initial IF decision matrix
    exmatrix = np.zeros((if_matrix.shape[0] + 2, if_matrix.shape[1]))
    exmatrix[:-2] = if_matrix

    for i in range(if_matrix.shape[1]):
        if types[i] == 1:
            exmatrix[-2, i] = np.max(if_matrix[:, i])
            exmatrix[-1, i] = np.min(if_matrix[:, i])
        else:
            exmatrix[-2, i] = np.min(if_matrix[:, i])
            exmatrix[-1, i] = np.max(if_matrix[:, i])

    # normalized matrix
    nmatrix = exmatrix.copy()
    nmatrix[:, types == 1] = exmatrix[:, types == 1] / np.max(exmatrix[:, types == 1])
    nmatrix[:, types == -1] = np.min(exmatrix[:, types == -1]) / exmatrix[:, types == -1]

    # weighted matrix
    wmatrix = nmatrix * weights

    # s matrix
    smatrix = np.sum(wmatrix, axis=1)

    # utility degree
    km = (smatrix / smatrix[-1])[:-2]
    kp = (smatrix / smatrix[-2])[:-2]

    # anti-ideal and ideal solutions utility functions
    fkm = kp / (kp + km)
    fkp = km / (kp + km)
    
    # final utility function
    f = (kp + km) / ( 1 + ((1 - fkp) / fkp) + ((1 - fkm) / fkm) )

    return f


