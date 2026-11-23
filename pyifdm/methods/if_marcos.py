# Copyright (c) 2022-2026 Jakub Więckowski & contributors

from __future__ import annotations

import numpy as np

from .base import IFMCDAMethod
from pyifdm.utils.score import chen_score_1


class ifMARCOS(IFMCDAMethod):
    """
    Intuitionistic Fuzzy MARCOS (Measurement of Alternatives and Ranking according to Compromise Solution).
    """

    descending = True
    requires_mixed_types = True

    def __init__(self) -> None:
        super().__init__()

    def _evaluate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(weights[0], (float, np.floating)):
            weights = chen_score_1(weights)

        if matrix.shape[2] == 2:
            new_matrix = np.zeros((matrix.shape[0], matrix.shape[1], matrix.shape[2] + 1))
            new_matrix[:, :, 0] = matrix[:, :, 0]
            new_matrix[:, :, 1] = matrix[:, :, 1]
            new_matrix[:, :, 2] = 1 - (matrix[:, :, 0] + matrix[:, :, 1])  
            matrix = new_matrix

        # aggregated IF decision matrix
        matrix_p =  np.sqrt((matrix[:, :, 0] - 1)**2 + (matrix[:, :, 1] - 0)**2 + (matrix[:, :, 2] - 0)**2)
        matrix_m =  np.sqrt((matrix[:, :, 0] - 0)**2 + (matrix[:, :, 1] - 1)**2 + (matrix[:, :, 2] - 0)**2)
        if_matrix = matrix_m / (matrix_m + matrix_p)

        # Extended initial IF decision matrix
        exmatrix = np.zeros((if_matrix.shape[0] + 2, if_matrix.shape[1]))
        exmatrix[:-2] = if_matrix

        for i in range(if_matrix.shape[1]):
            if types[i] == 1:
                exmatrix[-2, i] = np.max(if_matrix[:, i])
                exmatrix[-1, i] = np.min(if_matrix[:, i])
            else:
                exmatrix[-2, i] = np.min(if_matrix[:, i])
                exmatrix[-1, i] = np.max(if_matrix[:, i])

        # normalized matrix
        nmatrix = exmatrix.copy()
        nmatrix[:, types == 1] = exmatrix[:, types == 1] / np.max(exmatrix[:, types == 1])
        nmatrix[:, types == -1] = np.min(exmatrix[:, types == -1]) / exmatrix[:, types == -1]

        # weighted matrix
        wmatrix = nmatrix * weights

        # s matrix
        smatrix = np.sum(wmatrix, axis=1)

        # utility degree
        km = (smatrix / smatrix[-1])[:-2]
        kp = (smatrix / smatrix[-2])[:-2]

        # anti-ideal and ideal solutions utility functions
        fkm = kp / (kp + km)
        fkp = km / (kp + km)

        # final utility function
        f = (kp + km) / ( 1 + ((1 - fkp) / fkp) + ((1 - fkm) / fkm) )

        return f
