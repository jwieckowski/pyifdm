# Copyright (c) 2022-2026 Jakub Więckowski & contributors

from __future__ import annotations

from typing import Callable
import numpy as np

from .base import IFMCDAMethod
from pyifdm.utils.score import chen_score_1

class ifOCRA(IFMCDAMethod):
    """Intuitionistic Fuzzy OCRA (Operational Competitiveness Rating)."""

    descending = True
    requires_mixed_types = False

    def __init__(self, score: Callable = chen_score_1, logger=None) -> None:
        super().__init__(logger=logger)
        self.score = score

    def _evaluate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
    ) -> np.ndarray:
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1:
            w = np.asarray(self.score(w), dtype=float)
        self._log_step("weights", w, "Crisp criterion weights")

        smatrix = np.asarray(self.score(matrix), dtype=float)
        self._log_step("score_matrix", smatrix, "Crisp score of decision matrix")

        P_rows, Q_rows = [], []
        for j in range(smatrix.shape[1]):
            col = smatrix[:, j]
            span = np.max(col) - np.min(col)
            if span == 0:
                span = 1.0
            if types[j] == 1:
                P_rows.append([w[j] * ((col[i] - np.min(col)) / span) for i in range(smatrix.shape[0])])
            else:
                Q_rows.append([w[j] * ((np.max(col) - col[i]) / span) for i in range(smatrix.shape[0])])

        P = np.sum(np.array(P_rows), axis=0) if P_rows else np.zeros(smatrix.shape[0])
        Q = np.sum(np.array(Q_rows), axis=0) if Q_rows else np.zeros(smatrix.shape[0])
        self._log_step("P", P, "Aggregate performance for profit criteria")
        self._log_step("Q", Q, "Aggregate performance for cost criteria")

        P = P - np.min(P)
        Q = Q - np.min(Q)
        OPR = (P + Q) - np.min(P + Q)
        self._log_step("preferences", OPR, "Overall performance rating")
        
        return OPR.astype(float)