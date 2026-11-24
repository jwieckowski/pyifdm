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
        score: Callable = wan_dong_score_1,
        logger=None,
    ) -> None:
        super().__init__(logger=logger)
        self.normalization = normalization
        self.score = score

    def _evaluate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
    ) -> np.ndarray:
        # Optimal reference values per criterion
        R = np.zeros((matrix.shape[1], matrix.shape[2]), dtype=float)
        for j in range(matrix.shape[1]):
            if types[j] == 1:
                R[j] = matrix[np.argmax(matrix[:, j, 0]), j]
            else:
                R[j] = matrix[np.argmin(matrix[:, j, 0]), j]
        self._log_step(
            "optimal_reference",
            R,
            "Ideal criterion values selected from the decision matrix",
        )

        # Extended decision matrix (reference alternative in the first row)
        exmatrix = np.ones(
            (matrix.shape[0] + 1, matrix.shape[1], matrix.shape[2]),
            dtype=float,
        )
        exmatrix[0] = R
        exmatrix[1:] = matrix
        self._log_step(
            "extended_matrix",
            exmatrix,
            "Decision matrix extended with the optimal reference row",
        )

        # Normalized matrix
        nmatrix = np.asarray(self.normalization(exmatrix, types), dtype=float)
        self._log_step(
            "normalized_matrix",
            nmatrix,
            "Normalized extended decision matrix",
        )

        # Crisp or IFS weights → always use 2 components for weighting
        w = np.asarray(weights, dtype=float)
        if w.ndim == 1:
            w = np.repeat(w, 2).reshape((len(w), 2))
        self._log_step(
            "weights",
            w,
            "Criterion weights used for aggregation (IFS form)",
        )

        # Weighted normalized matrix
        wmatrix = np.zeros_like(nmatrix, dtype=float)
        wmatrix[:, :, 0] = 1 - (1 - nmatrix[:, :, 0]) ** w[:, 0]
        wmatrix[:, :, 1] = nmatrix[:, :, 1] ** w[:, 1]
        if nmatrix.shape[2] > 2:
            wmatrix[:, :, 2] = np.maximum(
                0.0, 1.0 - wmatrix[:, :, 0] - wmatrix[:, :, 1]
            )
        self._log_step(
            "weighted_matrix",
            wmatrix,
            "Weighted normalized decision matrix",
        )

        # Score values and overall performance
        S = np.asarray(self.score(wmatrix), dtype=float)
        self._log_step(
            "score_matrix",
            S,
            "Crisp score of each IFS entry in the weighted matrix",
        )

        M = np.sum(S, axis=1)
        self._log_step(
            "performance_index",
            M,
            "Overall performance index (row 0 = optimal reference)",
        )

        Q = M[1:] / M[0]
        self._log_step(
            "utility_degree",
            Q,
            "Utility degree of alternatives relative to the optimal reference",
        )
        return Q.astype(float)