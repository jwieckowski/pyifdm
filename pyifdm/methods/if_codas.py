# Copyright (c) 2022-2026 Jakub Więckowski & contributors

from __future__ import annotations

from typing import Callable

import numpy as np

from .base import IFMCDAMethod
from pyifdm.utils.normalization import swap_normalization
from pyifdm.utils.distance import euclidean_distance, hamming_distance


class ifCODAS(IFMCDAMethod):
    """Intuitionistic Fuzzy CODAS (Combinative Distance-based Assessment)."""

    descending = True
    requires_mixed_types = True

    def __init__(
        self,
        normalization: Callable = swap_normalization,
        distance_1: Callable = euclidean_distance,
        distance_2: Callable = hamming_distance,
        tau: float = 0.05,
        logger=None,
    ) -> None:
        super().__init__(logger=logger)
        self.normalization = normalization
        self.distance_1 = distance_1
        self.distance_2 = distance_2
        self.tau = float(tau)

    def _evaluate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
    ) -> np.ndarray:
        def _psi(val) -> int:
            return 1 if abs(float(np.asarray(val).reshape(-1)[0])) >= self.tau else 0

        def calculate_distance(method, row, Am, f):
            vals = [method(row[j], Am[j]) for j in range(row.shape[0])]
            if method.__name__ == "normalized_euclidean_distance":
                return np.sqrt(f * np.sum(vals))
            if method.__name__ == "normalized_hamming_distance":
                return f * np.sum(vals)
            return np.sum(vals)

        nmatrix = np.asarray(self.normalization(matrix, types), dtype=float)
        self._log_step("normalized_matrix", nmatrix, "Normalized decision matrix")

        w = np.asarray(weights, dtype=float)
        if w.ndim == 1:
            w = np.repeat(w, 2).reshape((len(w), 2))
        self._log_step("weights", w, "Criterion weights (IFS form)")

        wmatrix = np.zeros((nmatrix.shape[0], nmatrix.shape[1], 3), dtype=float)
        wmatrix[:, :, 0] = nmatrix[:, :, 0] * w[:, 0]
        wmatrix[:, :, 1] = nmatrix[:, :, 1] + w[:, 1] - nmatrix[:, :, 1] * w[:, 1]
        wmatrix[:, :, 2] = 1 - wmatrix[:, :, 0] - wmatrix[:, :, 1]
        self._log_step("weighted_matrix", wmatrix, "Weighted normalized matrix")

        Am = np.zeros((wmatrix.shape[1], 3), dtype=float)
        # profit criteria
        Am[types == 1, 0] = np.min(wmatrix[:, types == 1, 0], axis=0)
        Am[types == 1, 1] = np.max(wmatrix[:, types == 1, 1], axis=0)
        Am[types == 1, 2] = 1 - Am[types == 1, 0] - Am[types == 1, 1]
        # cost criteria
        Am[types == -1, 0] = np.max(wmatrix[:, types == -1, 0], axis=0)
        Am[types == -1, 1] = np.min(wmatrix[:, types == -1, 1], axis=0)
        Am[types == -1, 2] = 1 - Am[types == -1, 0] - Am[types == -1, 1]
        self._log_step("negative_ideal", Am, "Negative ideal solution")

        f1 = f2 = 1.0
        if "normalized" in self.distance_1.__name__:
            f1 = 1 / (2 * matrix.shape[1])
        if "normalized" in self.distance_2.__name__:
            f2 = 1 / (2 * matrix.shape[1])

        D1 = np.array(
            [calculate_distance(self.distance_1, wmatrix[i], Am, f1) for i in range(wmatrix.shape[0])],
            dtype=float,
        )
        D2 = np.array(
            [calculate_distance(self.distance_2, wmatrix[i], Am, f2) for i in range(wmatrix.shape[0])],
            dtype=float,
        )
        self._log_step("distance_1", D1, "Primary distance from negative ideal")
        self._log_step("distance_2", D2, "Secondary distance from negative ideal")

        RA = np.zeros((matrix.shape[0], matrix.shape[0]), dtype=float)
        for i in range(RA.shape[0]):
            for j in range(RA.shape[1]):
                RA[i, j] = (D1[i] - D1[j]) + (_psi(D1[i] - D1[j]) * (D2[i] - D2[j]))
        self._log_step("relative_assessment", RA, "Relative assessment matrix")

        AS = np.sum(RA, axis=1)
        self._log_step("preferences", AS, "Assessment scores")
        return AS.astype(float)