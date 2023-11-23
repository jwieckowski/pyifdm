# Copyright (c) 2023 Jakub Więckowski

import numpy as np
from pyifdm.methods.ifs.score import * 

def ifs(matrix, weights, types, score):
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

            score: callable
                Function used to calculate crisp score of IFS
            
        Returns
        -------
            ndarray
                Crisp preferences of alternatives

    """

    # score matrix
    smatrix = score(matrix)

    #  if performance rating
    P, Q = [], []
    for j in range(smatrix.shape[1]):
        if types[j] == 1:
            P.append([weights[j] * ((smatrix[i, j]) - np.min(smatrix[:, j]) / (np.max(smatrix[:, j]) - np.min(smatrix[:, j]))) for i in range(smatrix.shape[0])])
        else:
            Q.append([weights[j] * ((np.max(smatrix[:, j] - smatrix[i, j]) / (np.max(smatrix[:, j]) - np.min(smatrix[:, j]))))  for i in range(smatrix.shape[0])])

    P = np.sum(np.array(P), axis=0)
    Q = np.sum(np.array(Q), axis=0)

    # linear performance rating
    P -= np.min(P)
    Q -= np.min(Q)

    # overall performance rating
    OPR = (P + Q) - np.min(P + Q)
    return OPR
