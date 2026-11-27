# Copyright (c) 2022-2026 Jakub Więckowski & contributors

from __future__ import annotations

import numpy as np

from .base import IFMCDAMethod
from pyifdm.utils.score import chen_score_1


class ifMARCOS(IFMCDAMethod):
    """Intuitionistic Fuzzy MARCOS (Measurement of Alternatives and Ranking according to Compromise Solution)."""

    descending = True
    requires_mixed_types = True

    def __init__(self, logger=None) -> None:
        super().__init__(logger=logger)

    def _evaluate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
    ) -> np.ndarray:
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1:
            w = np.asarray(chen_score_1(w), dtype=float)
        self._log_step("weights", w, "Crisp criterion weights")

        mat = np.asarray(matrix, dtype=float)
        if mat.shape[2] == 2:
            new_matrix = np.zeros((mat.shape[0], mat.shape[1], 3), dtype=float)
            new_matrix[:, :, 0] = mat[:, :, 0]
            new_matrix[:, :, 1] = mat[:, :, 1]
            new_matrix[:, :, 2] = 1 - (mat[:, :, 0] + mat[:, :, 1])
            mat = new_matrix
        self._log_step("decision_matrix", mat, "Decision matrix with hesitation degree")

        matrix_p = np.sqrt(
            (mat[:, :, 0] - 1) ** 2 + (mat[:, :, 1] - 0) ** 2 + (mat[:, :, 2] - 0) ** 2
        )
        matrix_m = np.sqrt(
            (mat[:, :, 0] - 0) ** 2 + (mat[:, :, 1] - 1) ** 2 + (mat[:, :, 2] - 0) ** 2
        )
        if_matrix = matrix_m / (matrix_m + matrix_p)
        self._log_step("crisp_matrix", if_matrix, "Aggregated crisp IF decision matrix")

        exmatrix = np.zeros((if_matrix.shape[0] + 2, if_matrix.shape[1]), dtype=float)
        exmatrix[:-2] = if_matrix
        for i in range(if_matrix.shape[1]):
            if types[i] == 1:
                exmatrix[-2, i] = np.max(if_matrix[:, i])
                exmatrix[-1, i] = np.min(if_matrix[:, i])
            else:
                exmatrix[-2, i] = np.min(if_matrix[:, i])
                exmatrix[-1, i] = np.max(if_matrix[:, i])
        self._log_step(
            "extended_matrix",
            exmatrix,
            "Extended matrix with ideal and anti-ideal rows",
        )

        nmatrix = exmatrix.copy()
        nmatrix[:, types == 1] = exmatrix[:, types == 1] / np.max(exmatrix[:, types == 1], axis=0)
        nmatrix[:, types == -1] = np.min(exmatrix[:, types == -1], axis=0) / exmatrix[:, types == -1]
        self._log_step("normalized_matrix", nmatrix, "Normalized extended matrix")

        wmatrix = nmatrix * w
        self._log_step("weighted_matrix", wmatrix, "Weighted normalized matrix")

        smatrix = np.sum(wmatrix, axis=1)
        self._log_step("s_matrix", smatrix, "Row sums of weighted matrix")

        km = (smatrix / smatrix[-1])[:-2]
        kp = (smatrix / smatrix[-2])[:-2]
        self._log_step("utility_km", km, "Utility degree relative to anti-ideal")
        self._log_step("utility_kp", kp, "Utility degree relative to ideal")

        fkm = kp / (kp + km)
        fkp = km / (kp + km)
        f = (kp + km) / (1 + ((1 - fkp) / fkp) + ((1 - fkm) / fkm))
        self._log_step("preferences", f, "Final utility function values")
        
        return f.astype(float)