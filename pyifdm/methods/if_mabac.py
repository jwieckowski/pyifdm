# Copyright (c) 2022-2026 Jakub Więckowski & contributors

from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from .base import IFMCDAMethod
from pyifdm.utils.normalization import swap_normalization
from pyifdm.utils.score import liu_wang_score
from pyifdm.utils.distance import luo_distance

class ifMABAC(IFMCDAMethod):
    """Intuitionistic Fuzzy MABAC (Multi-Attributive Border Approximation area Comparison)."""

    descending = True
    requires_mixed_types = False

    def __init__(
        self,
        normalization: Callable = swap_normalization,
        distance: Callable = luo_distance,
        score: Callable = liu_wang_score,
        p: float = 2.25,
        g: float = 0.88,
        logger=None,
    ) -> None:
        super().__init__(logger=logger)
        self.normalization = normalization
        self.distance = distance
        self.score = score
        self.p = float(p)
        self.g = float(g)

    def _evaluate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
    ) -> np.ndarray:
        def calculate_distance(method, a, b, f):
            if method.__name__ == "normalized_euclidean_distance":
                return np.sqrt(f * method(a, b))
            if method.__name__ == "normalized_hamming_distance":
                return f * method(a, b)
            return method(a, b)

        nmatrix = np.asarray(self.normalization(matrix, types), dtype=float)
        self._log_step("normalized_matrix", nmatrix, "Normalized decision matrix")

        w = np.asarray(weights, dtype=float)
        if w.ndim == 1:
            w = np.repeat(w, 2).reshape((len(w), 2))
        self._log_step("weights", w, "Criterion weights (IFS form)")

        wmatrix = np.zeros((nmatrix.shape[0], nmatrix.shape[1], 3), dtype=float)
        wmatrix[:, :, 0] = nmatrix[:, :, 0] ** w[:, 0]
        wmatrix[:, :, 1] = 1 - (1 - nmatrix[:, :, 1]) ** w[:, 1]
        wmatrix[:, :, 2] = 1 - wmatrix[:, :, 0] - wmatrix[:, :, 1]
        self._log_step("weighted_matrix", wmatrix, "IFWG weighted matrix")

        G = np.zeros((wmatrix.shape[1], wmatrix.shape[2]), dtype=float)
        G[:, 0] = np.prod(wmatrix[:, :, 0], axis=0) ** (1 / wmatrix.shape[0])
        G[:, 1] = 1 - np.prod(1 - wmatrix[:, :, 1], axis=0) ** (1 / wmatrix.shape[0])
        G[:, 2] = np.maximum(0.0, 1 - G[:, 0] - G[:, 1])
        self._log_step("border_area", G, "Border approximation area")

        f = 1.0
        if "normalized" in self.distance.__name__:
            f = 1 / (2 * matrix.shape[1])

        DM = np.zeros((wmatrix.shape[0], wmatrix.shape[1]), dtype=float)
        for i in range(wmatrix.shape[0]):
            for j in range(wmatrix.shape[1]):
                dist = calculate_distance(self.distance, wmatrix[i, j], G[j], f)
                if self.score(wmatrix[i, j]) > self.score(G[j]):
                    DM[i, j] = dist ** self.g
                else:
                    DM[i, j] = -self.p * (dist ** self.g)
        self._log_step("discrimination_measures", DM, "Discrimination measure matrix")

        C = np.sum(DM, axis=1)
        self._log_step("preferences", C, "Assessment scores")
        return C.astype(float)