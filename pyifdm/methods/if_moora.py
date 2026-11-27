# Copyright (c) 2022-2026 Jakub Więckowski & contributors

from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from .base import IFMCDAMethod
from pyifdm.utils.score import zhang_xu_score_2

class ifMOORA(IFMCDAMethod):
    """Intuitionistic Fuzzy MOORA (Multi-Objective Optimization by Ratio Analysis)."""

    descending = True
    requires_mixed_types = True

    def __init__(
        self,
        score: Callable = zhang_xu_score_2,
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

        wmatrix = np.zeros((nmatrix.shape[0], nmatrix.shape[1], 3), dtype=float)
        wmatrix[:, :, 0] = nmatrix[:, :, 0] * w[:, 0]
        wmatrix[:, :, 1] = nmatrix[:, :, 1] + w[:, 1] - nmatrix[:, :, 1] * w[:, 1]
        wmatrix[:, :, 2] = 1 - wmatrix[:, :, 1] - wmatrix[:, :, 0]
        self._log_step("weighted_matrix", wmatrix, "Weighted normalized matrix")

        Sp = np.zeros((matrix.shape[0], 3), dtype=float)
        Sm = np.zeros((matrix.shape[0], 3), dtype=float)
        profit_indexes = np.where(types == 1)[0]
        cost_indexes = np.where(types == -1)[0]

        for i in range(wmatrix.shape[0]):
            profit = wmatrix[i, profit_indexes[0]].copy()
            cost = wmatrix[i, cost_indexes[0]].copy()
            for idx in profit_indexes[1:]:
                profit[0] = profit[0] + wmatrix[i, idx, 0] - profit[0] * wmatrix[i, idx, 0]
                profit[1] *= wmatrix[i, idx, 1]
            for idx in cost_indexes[1:]:
                cost[0] = cost[0] + wmatrix[i, idx, 0] - cost[0] * wmatrix[i, idx, 0]
                cost[1] *= wmatrix[i, idx, 1]
            Sp[i] = [profit[0], profit[1], 1 - profit[0] - profit[1]]
            Sm[i] = [cost[0], cost[1], 1 - cost[0] - cost[1]]

        self._log_step("benefit_sum", Sp, "Aggregated benefit IFS")
        self._log_step("cost_sum", Sm, "Aggregated cost IFS")

        Dp = np.asarray(self.score(Sp), dtype=float)
        Dm = np.asarray(self.score(Sm), dtype=float)
        prefs = Dp - Dm
        self._log_step("preferences", prefs, "Benefit score minus cost score")
        
        return prefs.astype(float)