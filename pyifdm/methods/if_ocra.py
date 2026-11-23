# Copyright (c) 2022-2026 Jakub Więckowski & contributors

from __future__ import annotations

from typing import Callable
import numpy as np

from .base import IFMCDAMethod
from pyifdm.utils.score import chen_score_1

class ifOCRA(IFMCDAMethod):
    """
    Intuitionistic Fuzzy OCRA (Operational Competitiveness Rating).
    """

    descending = True
    requires_mixed_types = False

    def __init__(
        self,
        score: Callable = chen_score_1
    ) -> None:
        super().__init__()
        self.score = score

    def _evaluate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(weights[0], (float, np.floating)):
            weights = self.score(weights)

        # score matrix
        smatrix = self.score(matrix)

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
