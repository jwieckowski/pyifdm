# Copyright (c) 2022-2026 Jakub Więckowski & contributors

from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from .base import IFMCDAMethod
from pyifdm.utils.score import zhang_xu_score_2

class ifMOORA(IFMCDAMethod):
    """
    Intuitionistic Fuzzy MOORA (Multi-Objective Optimization by Ratio Analysis).
    """

    descending = True
    requires_mixed_types = True

    def __init__(
        self,
        score: Callable = zhang_xu_score_2,
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
        wmatrix = np.zeros((nmatrix.shape[0], nmatrix.shape[1], 3), dtype=object)

        # crisp weights
        if weights.ndim == 1:
            weights = np.repeat(weights, 2).reshape((len(weights), 2))

        wmatrix[:, :, 0] = nmatrix[:, :, 0] * weights[:, 0]
        wmatrix[:, :, 1] = nmatrix[:, :, 1] + weights[:, 1] - nmatrix[:, :, 1] * weights[:, 1]
        wmatrix[:, :, 2] = 1 - wmatrix[:, :, 1] - wmatrix[:, :, 0]

        # sum of costs and benefits
        Sp, Sm  = np.zeros((matrix.shape[0], 3)), np.zeros((matrix.shape[0], 3))
        profit_indexes = np.where(types == 1)
        cost_indexes = np.where(types == -1)
        for i in range(wmatrix.shape[0]):
            profit = wmatrix[i, profit_indexes[0][0]]
            cost = wmatrix[i, cost_indexes[0][0]]

            for idx in profit_indexes[0][1:]:
                profit[0] = profit[0] + wmatrix[i, idx, 0] - profit[0] * wmatrix[i, idx, 0]
                profit[1] *= wmatrix[i, idx, 1]

            for idx in cost_indexes[0][1:]:
                cost[0] = cost[0] + wmatrix[i, idx, 0] - cost[0] * wmatrix[i, idx, 0]
                cost[1] *= wmatrix[i, idx, 1]

            # Sp.append([profit[0], profit[1], 1 - profit[0] - profit[1]])
            # Sm.append([cost[0], cost[1], 1 - cost[0] - cost[1]])
            Sp[i] = [profit[0], profit[1], 1 - profit[0] - profit[1]]
            Sm[i] = [cost[0], cost[1], 1 - cost[0] - cost[1]]

        # Sp, Sm = np.array(Sp), np.array(Sm) 

        # self.score functions
        Dp = self.score(Sp)
        Dm = self.score(Sm)

        return Dp - Dm
