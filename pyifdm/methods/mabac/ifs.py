# Copyright (c) 2022 Jakub Więckowski

import numpy as np

def ifs(matrix, weights, types, normalization, distance, score,p, g):
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

            p: float
                Adjust parameter for distance calculation

            g: float
                Adjust parameter for distance calculation
        
        Returns
        -------
            ndarray
                Crisp preferences of alternatives

    """

    def calculate_distance(method, wmatrix, G, f):
        """
            Calculates the distance between two Intuitionistic Fuzzy Sets (u, v) using given distance measure

            Parameters
            ----------
                method : callable
                    Function used to calculate distance

                wmatrix : ndarray
                    Weighted intuitionistic fuzzy matrix

                G : ndarray
                    Border approximation area
                
                f : float
                    Multiplication factor

            Returns
            -------
                float
                    Crisp value representing distance
        """
        if method.__name__ == 'normalized_euclidean_distance':
            return np.sqrt(f * method(wmatrix, G))
        elif method.__name__ == 'normalized_hamming_distance':
            return f * method(wmatrix, G)
        else:
            return method(wmatrix, G)

    # normalized matrix
    nmatrix = normalization(matrix, types)

    # weighted matrix
    wmatrix = np.zeros((nmatrix.shape[0], nmatrix.shape[1], 3), dtype=object)
    
    if weights.ndim == 1:
        weights = np.repeat(weights, 2).reshape((len(weights), 2))

    # intuitionistic fuzzy weighted geometric (IFWG) operator
    wmatrix[:, :, 0] = nmatrix[:, :, 0]**weights[:, 0]
    wmatrix[:, :, 1] = 1 - (1 - nmatrix[:, :, 1])**weights[:, 1]
    wmatrix[:, :, 2] = 1 - wmatrix[:, :, 0] - wmatrix[:, :, 1]

    # border approximation area
    G = np.zeros((wmatrix.shape[1], wmatrix.shape[2]), dtype=object)
    G[:, 0] = np.prod(wmatrix[:, :, 0], axis=0)**(1/wmatrix.shape[0])
    G[:, 1] = 1 - np.prod(1 - wmatrix[:, :, 1], axis=0)**(1/wmatrix.shape[0])

    f = 1
    if 'normalized' in distance.__name__:
        f = 1/(2*matrix.shape[1])

    # discrimination measures
    DM = np.zeros((wmatrix.shape[0], wmatrix.shape[1]))
    for i in range(wmatrix.shape[0]):
        for j in range(wmatrix.shape[1]):
            if score(wmatrix[i,j]) > score(G[j]):
                DM[i,j] = calculate_distance(distance, wmatrix[i,j], G[j], f)**g
            else:
                DM[i, j] = -p * calculate_distance(distance, wmatrix[i,j], G[j], f)**g

    # assessment score
    C = np.sum(DM, axis=1)
    return C
