# Copyright (c) 2022 Jakub Więckowski

import numpy as np

def ifs(matrix, weights, types, normalization, distance_1, distance_2, tau):
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

            distance_1: callable
                Function used to calculate distance between two IFS

            distance_2: callable
                Function used to calculate distance between two IFS

            tau: float
                Threshold parameter

        Returns
        -------
            ndarray
                Crisp preferences of alternatives

    """
    def _psi(ifs, tau=tau):
        # tau from 0.01 to 0.05
        if np.abs(ifs) >= tau:
            return 1
        return 0

    def calculate_distance(method, wmatrix, Am, f):
        """
            Calculates the distance between two Intuitionistic Fuzzy Sets (u, v) using given distance measure

            Parameters
            ----------
                method : callable
                    Function used to calculate distance

                wmatrix : ndarray
                    Weighted intuitionistic fuzzy matrix

                Am : ndarray
                    Negative ideal solution
                
                f : float
                    Multiplication factor

            Returns
            -------
                float
                    Crisp value representing distance
        """
        if method.__name__ == 'normalized_euclidean_distance':
            return np.sqrt(f * np.sum([method(wmatrix[j], Am[j])
                            for j in range(wmatrix.shape[0])]))
        elif method.__name__ == 'normalized_hamming_distance':
            return f * np.sum([method(wmatrix[j], Am[j])
                            for j in range(wmatrix.shape[0])])
        else:
            return np.sum([method(wmatrix[j], Am[j])
                            for j in range(wmatrix.shape[0])])

    # normalized matrix
    nmatrix = normalization(matrix, types)

    # weighted normalized matrix
    wmatrix = np.zeros((nmatrix.shape[0], nmatrix.shape[1], 3), dtype=object)
    
    # transform crisp weights to ifs
    if weights.ndim == 1:
        weights = np.repeat(weights, 2).reshape((len(weights), 2))

    wmatrix[:, :, 0] = nmatrix[:, :, 0] * weights[:, 0]
    wmatrix[:, :, 1] = nmatrix[:, :, 1] + weights[:, 1] - nmatrix[:, :, 1] * weights[:, 1]
    wmatrix[:, :, 2] = 1 - wmatrix[:, :, 0] - wmatrix[:, :, 1]

    # negative ideal solution
    Am = np.zeros((wmatrix.shape[1], 3), dtype=object)
    # profit criteria
    Am[types==1, 0] = np.min(wmatrix[:, types==1, 0], axis=0)
    Am[types==1, 1] = np.max(wmatrix[:, types==1, 1], axis=0)
    Am[types==1, 2] = 1 - Am[types==1, 0] - Am[types==1, 1]
    # cost criteria
    Am[types==-1, 0] = np.max(wmatrix[:, types==-1, 0], axis=0)
    Am[types==-1, 1] = np.min(wmatrix[:, types==-1, 1], axis=0)
    Am[types==-1, 2] = 1 - Am[types==-1, 0] - Am[types==-1, 1]

    f1, f2 = 1, 1
    if 'normalized' in distance_1.__name__:
        f1 = 1/(2*matrix.shape[1])
    if 'normalized' in distance_2.__name__:
        f2 = 1/(2*matrix.shape[1])

    # distances
    D1, D2 = np.zeros((wmatrix.shape[0], 1), dtype=object), np.zeros((wmatrix.shape[0], 1), dtype=object)
    for i in range(wmatrix.shape[0]):
        D1[i] = calculate_distance(distance_1, wmatrix[i], Am, f1)
        D2[i] = calculate_distance(distance_2, wmatrix[i], Am, f2)
        
    # relative assessment matrix
    RA = np.zeros((matrix.shape[0], matrix.shape[0]))
    for i in range(RA.shape[0]):
        for j in range(RA.shape[1]):
            RA[i, j] = (D1[i] - D1[j]) + \
                (_psi(D1[i] - D1[j]) * (D2[i] - D2[j]))

    # assessment score
    AS = np.sum(RA, axis=1)
    return AS
