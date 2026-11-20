# Copyright (c) 2022-2026 Jakub Więckowski & contributors
from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from .base import IFMCDAMethod
from pyifdm.utils.normalization import swap_normalization
from pyifdm.utils.score import wan_dong_score_1

class ifARAS(IFMCDAMethod):
    """
    Intuitionistic Fuzzy ARAS (Additive Ratio Assessment).
    """

    descending = True
    requires_mixed_types = False

    def __init__(
        self,
        normalization: Optional[Callable] = swap_normalization,
        score: Callable = wan_dong_score_1
    ) -> None:
        super().__init__()
        self.normalization = normalization
        self.score = score

    def _evaluate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
    ) -> np.ndarray:
        R = np.zeros((matrix.shape[1], matrix.shape[2]), dtype=object)
        for j in range(matrix.shape[1]):
            if types[j] == 1:
                R[j] = matrix[np.argmax(matrix[:, j, 0]), j]
            else:
                R[j] = matrix[np.argmin(matrix[:, j, 0]), j]

        # extended decision matrix
        exmatrix = np.ones((matrix.shape[0]+1, matrix.shape[1], matrix.shape[2]), dtype=object)
        exmatrix[0] = R
        exmatrix[1:] = matrix

        # normalized matrix
        nmatrix = self.normalization(exmatrix, types)

        # weighted normalized matrix
        wmatrix = np.zeros((nmatrix.shape[0], nmatrix.shape[1], nmatrix.shape[2]), dtype=object)

        # crisp weights
        if weights.ndim == 1:
            weights = np.repeat(weights, 2).reshape((len(weights), 2))

        wmatrix[:, :, 0] = 1 - (1 - nmatrix[:, :, 0])**weights[:, 0]
        wmatrix[:, :, 1] = nmatrix[:, :, 1]**weights[:, 1]

        # self.score values
        S = self.score(wmatrix)

        # overal performance rating
        M = np.sum(S, axis=1)

        Q = M[1:] / M[0]
        return Q