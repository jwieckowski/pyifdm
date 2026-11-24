# Copyright (c) 2022-2026 Jakub Więckowski & contributors

from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from .base import IFMCDAMethod
from pyifdm.utils.normalization import swap_normalization
from pyifdm.utils.score import liu_wang_score

class ifEDAS(IFMCDAMethod):
    """Intuitionistic Fuzzy EDAS (Evaluation based on Distance from Average Solution)."""

    descending = True
    requires_mixed_types = False

    def __init__(
        self,
        normalization: Callable = swap_normalization,
        score: Callable = liu_wang_score,
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
        nmatrix = np.asarray(self.normalization(matrix, types), dtype=float)
        self._log_step("normalized_matrix", nmatrix, "Normalized decision matrix")

        av = np.array(
            [
                [
                    1 - (np.prod(1 - nmatrix[:, j, 0])) ** (1 / nmatrix.shape[0]),
                    (np.prod(nmatrix[:, j, 1])) ** (1 / nmatrix.shape[0]),
                ]
                for j in range(nmatrix.shape[1])
            ],
            dtype=float,
        )
        self._log_step("average_solution", av, "Average solution per criterion")

        pda = np.zeros((nmatrix.shape[0], nmatrix.shape[1]), dtype=float)
        nda = np.zeros((nmatrix.shape[0], nmatrix.shape[1]), dtype=float)
        for i in range(nmatrix.shape[0]):
            for j in range(nmatrix.shape[1]):
                s_ij = self.score(nmatrix[i, j])
                s_av = self.score(av[j])
                pda[i, j] = np.max([0.0, (s_ij - s_av)]) / s_av
                nda[i, j] = np.max([0.0, (s_av - s_ij)]) / s_av
        self._log_step("pda", pda, "Positive distance from average")
        self._log_step("nda", nda, "Negative distance from average")

        w = np.asarray(weights, dtype=float)
        if w.ndim == 2:
            w = np.array([self.score(x) for x in w], dtype=float)
        self._log_step("weights", w, "Crisp criterion weights")

        sp = np.sum(w * pda, axis=1)
        sn = np.sum(w * nda, axis=1)
        self._log_step("sp", sp, "Weighted sum of positive distances")
        self._log_step("sn", sn, "Weighted sum of negative distances")

        nsp = sp / np.max(sp) if np.max(sp) != 0 else np.zeros_like(sp)
        nsn = 1 - sn / np.max(sn) if np.max(sn) != 0 else np.zeros_like(sn)
        self._log_step("nsp", nsp, "Normalized SP")
        self._log_step("nsn", nsn, "Normalized SN")

        appraisal = 0.5 * (nsp + nsn)
        self._log_step("appraisal_score", appraisal, "Final appraisal score")
        return appraisal.astype(float)