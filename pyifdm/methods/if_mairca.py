# Copyright (c) 2022-2026 Jakub Więckowski & contributors

from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from .base import IFMCDAMethod
from pyifdm.utils.normalization import minmax_normalization
from pyifdm.utils.distance import normalized_euclidean_distance
from pyifdm.utils.score import liu_wang_score

class ifMAIRCA(IFMCDAMethod):
    """
    Intuitionistic Fuzzy MAIRCA (MultiAttributive Ideal-Real Comparative Analysis).
    """

    descending = True
    requires_mixed_types = False

    def __init__(
        self,
        normalization: Callable = minmax_normalization,
        distance: Callable = normalized_euclidean_distance,
        score: Callable = liu_wang_score
    ) -> None:
        super().__init__()
        self.normalization = normalization
        self.distance = distance
        self.score = score

    def _evaluate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
    ) -> np.ndarray:
        f = 1
        if 'normalized' in self.distance.__name__:
            f = 1/(2*matrix.shape[1])

        # self.distance measures
        dm = np.zeros((matrix.shape[0], matrix.shape[1], 2), dtype=object)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if self.distance.__name__ == 'normalized_euclidean_distance':
                    dm[i, j, 0] = np.sqrt(f * self.distance(matrix[i, j], np.array([1, 0, 0])))
                    dm[i, j, 1] = np.sqrt(f * self.distance(matrix[i, j], np.array([0, 1, 0])))
                elif self.distance.__name__ == 'normalized_hamming_distance':
                    dm[i, j, 0] = f * self.distance(matrix[i, j], np.array([1, 0, 0]))
                    dm[i, j, 1] = f * self.distance(matrix[i, j], np.array([0, 1, 0]))
                else:
                    dm[i, j, 0] = self.distance(matrix[i, j], np.array([1, 0, 0]))
                    dm[i, j, 1] = self.distance(matrix[i, j], np.array([0, 1, 0]))

        if self.normalization.__name__ != 'minmax_normalization':
            dm =  self.normalization(dm, types)

        # closeness coefficient
        cw = dm[:, :, 1] / (dm[:, :, 1] + dm[:, :, 0])

        # normalized matrix
        if self.normalization.__name__ == 'minmax_normalization':
            nmatrix = self.normalization(cw, types)
        else:
            nmatrix = cw

        # crisp weights
        if weights.ndim == 2:
            weights = np.array([self.score(w) for w in weights])

        # theoretical intuitionistic fuzzy decision matrix
        tdm = 1 / matrix.shape[0] * nmatrix * weights
        tdm = np.tile(np.max(tdm, axis=0), nmatrix.shape[0]).reshape((nmatrix.shape[0], nmatrix.shape[1]))

        # real evaluation matrix
        rem = nmatrix * tdm

        # gap matrix
        gm = tdm - rem

        # utility self.score
        s = np.sum(gm, axis=1)
        return s
