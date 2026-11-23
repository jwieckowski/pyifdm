# Copyright (c) 2022-2026 Jakub Więckowski & contributors

from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from .base import IFMCDAMethod
from pyifdm.utils.normalization import swap_normalization
from pyifdm.utils.score import liu_wang_score
from pyifdm.utils.distance import luo_distance

class ifMABAC(IFMCDAMethod):
    """
    Intuitionistic Fuzzy MABAC (Multi-Attributive Border Approximation area Comparison).
    """

    descending = True
    requires_mixed_types = False

    def __init__(
        self,
        normalization: Callable = swap_normalization,
        distance: Callable = luo_distance,
        score: Callable = liu_wang_score,
        p: float = 2.25,
        g: float = 0.88
    ) -> None:
        super().__init__()
        self.normalization = normalization
        self.distance = distance
        self.score = score
        self.p = float(p)
        self.g = float(g)

    def _evaluate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
    ) -> np.ndarray:
        def calculate_distance(method, wmatrix, G, f):
            """
                Calculates the self.distance between two Intuitionistic Fuzzy Sets (u, v) using given self.distance measure

                Parameters
                ----------
                    method : callable
                        Function used to calculate self.distance

                    wmatrix : ndarray
                        Weighted intuitionistic fuzzy matrix

                    G : ndarray
                        Border approximation area

                    f : float
                        Multiplication factor

                Returns
                -------
                    float
                        Crisp value representing self.distance
            """
            if method.__name__ == 'normalized_euclidean_distance':
                return np.sqrt(f * method(wmatrix, G))
            elif method.__name__ == 'normalized_hamming_distance':
                return f * method(wmatrix, G)
            else:
                return method(wmatrix, G)

        # normalized matrix
        nmatrix = self.normalization(matrix, types)

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
        if 'normalized' in self.distance.__name__:
            f = 1/(2*matrix.shape[1])

        # discrimination measures
        DM = np.zeros((wmatrix.shape[0], wmatrix.shape[1]))
        for i in range(wmatrix.shape[0]):
            for j in range(wmatrix.shape[1]):
                if self.score(wmatrix[i,j]) > self.score(G[j]):
                    DM[i,j] = calculate_distance(self.distance, wmatrix[i,j], G[j], f)**self.g
                else:
                    DM[i, j] = -self.p * calculate_distance(self.distance, wmatrix[i,j], G[j], f)**self.g

        # assessment self.score
        C = np.sum(DM, axis=1)
        return C
