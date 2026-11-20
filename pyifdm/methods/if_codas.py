# Copyright (c) 2022-2026 Jakub Więckowski & contributors

from __future__ import annotations

from typing import Callable

import numpy as np

from .base import IFMCDAMethod
from pyifdm.utils.normalization import swap_normalization
from pyifdm.utils.distance import euclidean_distance, hamming_distance


class ifCODAS(IFMCDAMethod):
    """
    Intuitionistic Fuzzy CODAS (Combinative Distance-based Assessment).
    """

    descending = True
    requires_mixed_types = True

    def __init__(
        self,
        normalization: Callable = swap_normalization,
        distance_1: Callable = euclidean_distance,
        distance_2: Callable = hamming_distance,
        tau: float = 0.05,
    ) -> None:
        super().__init__()
        self.normalization = normalization
        self.distance_1 = distance_1
        self.distance_2 = distance_2
        self.tau = float(tau)

    def _evaluate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
    ) -> np.ndarray:
        def _psi(val):
            # tau from 0.01 to 0.05
            if np.abs(val) >= self.tau:
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
        nmatrix = self.normalization(matrix, types)

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
        if 'normalized' in self.distance_1.__name__:
            f1 = 1/(2*matrix.shape[1])
        if 'normalized' in self.distance_2.__name__:
            f2 = 1/(2*matrix.shape[1])

        # distances
        D1, D2 = np.zeros((wmatrix.shape[0], 1), dtype=object), np.zeros((wmatrix.shape[0], 1), dtype=object)
        for i in range(wmatrix.shape[0]):
            D1[i] = calculate_distance(self.distance_1, wmatrix[i], Am, f1)
            D2[i] = calculate_distance(self.distance_2, wmatrix[i], Am, f2)

        # relative assessment matrix
        RA = np.zeros((matrix.shape[0], matrix.shape[0]))
        for i in range(RA.shape[0]):
            for j in range(RA.shape[1]):
                RA[i, j] = (D1[i] - D1[j]) + \
                    (_psi(D1[i] - D1[j]) * (D2[i] - D2[j]))

        # assessment score
        AS = np.sum(RA, axis=1)
        return AS
