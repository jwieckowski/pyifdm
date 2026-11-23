# Copyright (c) 2022-2026 Jakub Więckowski & contributors

from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from .base import IFMCDAMethod
from pyifdm.utils.distance import normalized_euclidean_distance

class ifTOPSIS(IFMCDAMethod):
    """
    Intuitionistic Fuzzy TOPSIS.
    """

    descending = True
    requires_mixed_types = True

    def __init__(
        self,
        distance: Callable = normalized_euclidean_distance,
        normalization: Optional[Callable] = None
    ) -> None:
        super().__init__()
        self.distance = distance
        self.normalization = normalization

    def _evaluate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
    ) -> np.ndarray:
        if self.normalization is not None:
            nmatrix = self.normalization(matrix, types)
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
        if 'normalized' in self.distance.__name__:
            f = 1/(2*matrix.shape[1])

        # self.distance from ideal solution
        splus, sminus = np.zeros((wmatrix.shape[0], )), np.zeros((wmatrix.shape[0], ))
        for i in range(wmatrix.shape[0]):
            if self.distance.__name__== 'normalized_euclidean_distance':
                splus[i] = np.sqrt(f * np.sum([self.distance(wmatrix[i, idx], ap) for idx, ap in enumerate(aplus)]))
                sminus[i] = np.sqrt(f * np.sum([self.distance(wmatrix[i, idx], am) for idx, am in enumerate(aminus)]))
            elif self.distance.__name__== 'normalized_hamming_distance':
                splus[i] = f * np.sum([self.distance(wmatrix[i, idx], ap) for idx, ap in enumerate(aplus)])
                sminus[i] = f * np.sum([self.distance(wmatrix[i, idx], am) for idx, am in enumerate(aminus)])
            else:
                splus[i] = np.sum([self.distance(wmatrix[i, idx], ap) for idx, ap in enumerate(aplus)])
                sminus[i] = np.sum([self.distance(wmatrix[i, idx], am) for idx, am in enumerate(aminus)])

        # assessment score
        return sminus / (splus + sminus)
