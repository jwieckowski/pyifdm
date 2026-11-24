# Copyright (c) 2022-2026 Jakub Więckowski & contributors

from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from .base import IFMCDAMethod
from pyifdm.utils.score import thakur_score

class ifCOPRAS(IFMCDAMethod):
    """Intuitionistic Fuzzy COPRAS (Complex Proportional Assessment)."""

    descending = True
    requires_mixed_types = False

    def __init__(
        self,
        score: Callable = thakur_score,
        normalization: Optional[Callable] = None,
        logger=None,
    ) -> None:
        super().__init__(logger=logger)
        self.score = score
        self.normalization = normalization

    def _evaluate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
    ) -> np.ndarray:
        if self.normalization is not None:
            nmatrix = np.asarray(self.normalization(matrix, types), dtype=float)
        else:
            nmatrix = np.asarray(matrix, dtype=float).copy()
        self._log_step("normalized_matrix", nmatrix, "Normalized decision matrix")

        w = np.asarray(weights, dtype=float)
        if w.ndim == 1:
            w = np.repeat(w, 2).reshape((len(w), 2))
        self._log_step("weights", w, "Criterion weights (IFS form)")

        wmatrix = np.zeros(nmatrix.shape, dtype=float)
        wmatrix[:, :, 0] = np.sqrt(1 - (1 - nmatrix[:, :, 0] ** 2) ** w[:, 0])
        wmatrix[:, :, 1] = np.sqrt((nmatrix[:, :, 1] ** 2) ** w[:, 1])
        self._log_step("weighted_matrix", wmatrix, "Weighted normalized matrix")

        s = np.asarray(self.score(wmatrix), dtype=float)
        self._log_step("score_matrix", s, "Crisp scores of weighted IFS")

        profit = types == 1
        cost = types == -1
        Sp = np.sum(s[:, profit], axis=1) / max(profit.sum(), 1)
        Sr = np.sum(s[:, cost], axis=1) / max(cost.sum(), 1)
        self._log_step("Sp", Sp, "Maximizing index")
        self._log_step("Sr", Sr, "Minimizing index")

        N = np.sum(np.exp(Sr)) / np.sum(1 / np.exp(Sr))
        Q = Sp + (N / np.exp(Sr))
        prefs = Q / np.max(Q)
        self._log_step("preferences", prefs, "Relative significance of alternatives")
        return prefs.astype(float)