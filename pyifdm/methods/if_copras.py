# Copyright (c) 2022-2026 Jakub Więckowski & contributors

from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from .base import IFMCDAMethod
from pyifdm.utils.score import thakur_score

class ifCOPRAS(IFMCDAMethod):
    """
    Intuitionistic Fuzzy COPRAS (Complex Proportional Assessment).
    """

    descending = True
    requires_mixed_types = False

    def __init__(
        self,
        score: Callable = thakur_score,
        normalization: Optional[Callable] = None
    ) -> None:
        super().__init__()
        self.score = score
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
        wmatrix = np.zeros(nmatrix.shape)

        # crisp weights
        if weights.ndim == 1:
            weights = np.repeat(weights, 2).reshape((len(weights), 2))

        wmatrix[:, :, 0] = np.sqrt(1 - (1 - nmatrix[:, :, 0] ** 2) ** weights[:, 0])
        wmatrix[:, :, 1] = np.sqrt((nmatrix[:, :, 1] ** 2) ** weights[:, 1])

        # Score function
        s = self.score(wmatrix)

        # Determine the maximizing and minimizing index
        Sp = np.sum(s[:, types == 1], axis=1) / types[types == 1].shape
        Sr = np.sum(s[:, types == -1], axis=1) / types[types == -1].shape

        # Determine the relative significance value of each alternative
        N = np.sum(np.exp(Sr)) / np.sum(1 / np.exp(Sr))
        Q =  Sp + (N / np.exp(Sr))

        return Q / np.max(Q)