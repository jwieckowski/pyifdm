# Copyright (c) 2022-2026 Jakub Więckowski & contributors

from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from .base import IFMCDAMethod
from pyifdm.utils.normalization import minmax_normalization
from pyifdm.utils.distance import normalized_euclidean_distance
from pyifdm.utils.score import liu_wang_score

class ifMAIRCA(IFMCDAMethod):
    """Intuitionistic Fuzzy MAIRCA (MultiAttributive Ideal-Real Comparative Analysis)."""

    descending = True
    requires_mixed_types = False

    def __init__(
        self,
        normalization: Callable = minmax_normalization,
        distance: Callable = normalized_euclidean_distance,
        score: Callable = liu_wang_score,
        logger=None,
    ) -> None:
        super().__init__(logger=logger)
        self.normalization = normalization
        self.distance = distance
        self.score = score

    def _evaluate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
    ) -> np.ndarray:
        f = 1.0
        if "normalized" in self.distance.__name__:
            f = 1 / (2 * matrix.shape[1])

        dm = np.zeros((matrix.shape[0], matrix.shape[1], 2), dtype=float)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ideal = np.array([1.0, 0.0, 0.0])
                anti = np.array([0.0, 1.0, 0.0])
                d_pos = self.distance(matrix[i, j], ideal)
                d_neg = self.distance(matrix[i, j], anti)
                if self.distance.__name__ == "normalized_euclidean_distance":
                    dm[i, j, 0] = np.sqrt(f * d_pos)
                    dm[i, j, 1] = np.sqrt(f * d_neg)
                elif self.distance.__name__ == "normalized_hamming_distance":
                    dm[i, j, 0] = f * d_pos
                    dm[i, j, 1] = f * d_neg
                else:
                    dm[i, j, 0] = d_pos
                    dm[i, j, 1] = d_neg
        self._log_step("distance_measures", dm, "Distances to ideal and anti-ideal")

        if self.normalization.__name__ != "minmax_normalization":
            dm = np.asarray(self.normalization(dm, types), dtype=float)
            self._log_step("normalized_distances", dm, "Normalized distance measures")

        cw = dm[:, :, 1] / (dm[:, :, 1] + dm[:, :, 0])
        self._log_step("closeness_coefficient", cw, "Closeness coefficient matrix")

        if self.normalization.__name__ == "minmax_normalization":
            nmatrix = np.asarray(self.normalization(cw, types), dtype=float)
        else:
            nmatrix = cw
        self._log_step("normalized_matrix", nmatrix, "Normalized closeness matrix")

        w = np.asarray(weights, dtype=float)
        if w.ndim == 2:
            w = np.array([self.score(x) for x in w], dtype=float)
        self._log_step("weights", w, "Crisp criterion weights")

        tdm = np.ones((nmatrix.shape[0], nmatrix.shape[1]), dtype=float)
        tdm = tdm * (1 / nmatrix.shape[0]) * w
        self._log_step("theoretical_matrix", tdm, "Theoretical decision matrix")

        rdm = nmatrix * tdm
        self._log_step("real_matrix", rdm, "Real decision matrix")

        gap = tdm - rdm
        self._log_step("gap_matrix", gap, "Gap matrix (theoretical - real)")

        prefs = np.sum(gap, axis=1)
        self._log_step("preferences", prefs, "Total gap scores")
        
        return prefs.astype(float)