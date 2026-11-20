# Copyright (c) 2022-2026 Jakub Więckowski & contributors

from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from .base import IFMCDAMethod
from pyifdm.utils.normalization import swap_normalization
from pyifdm.utils.score import liu_wang_score

class ifEDAS(IFMCDAMethod):
    """
    Intuitionistic Fuzzy EDAS (Evaluation based on Distance from Average Solution).
    """

    descending = True
    requires_mixed_types = False

    def __init__(
        self,
        normalization: Callable = swap_normalization,
        score: Callable = liu_wang_score
    ) -> None:
        super().__init__()
        self.normalization = normalization
        self.score = score

    def _evaluate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
    ) -> np.ndarray:
        nmatrix = self.normalization(matrix, types)

        # average solution
        av = np.array([[1 - (np.prod(1 - nmatrix[:, j, 0]))**(1/nmatrix.shape[0]),
                      (np.prod(nmatrix[:, j, 1]))**(1/nmatrix.shape[0])] for j in range(nmatrix.shape[1])])

        # positive and negative distances from average
        pda, nda = np.zeros((nmatrix.shape[0], nmatrix.shape[1])), np.zeros((nmatrix.shape[0], nmatrix.shape[1]))
        for i in range(nmatrix.shape[0]):
            for j in range(nmatrix.shape[1]):
                pda[i, j] = np.max([0, (self.score(nmatrix[i, j]) - self.score(av[j]))]) / self.score(av[j])
                nda[i, j] = np.max([0, (self.score(av[j]) - self.score(nmatrix[i, j]))]) / self.score(av[j])

        # crisp weights
        if weights.ndim == 2:
            weights = np.array([self.score(w) for w in weights])

        # weighted positive and negative distances
        sp = np.sum(weights * pda, axis=1)
        sn = np.sum(weights * nda, axis=1)

        # normalized weighted positive and negative distances
        nsp = sp / np.max(sp) if np.max(sp) != 0 else np.zeros((sp.shape))
        nsn = 1 - sn / np.max(sn) if np.max(sn) != 0 else np.zeros((sn.shape))

        # appraisal score
        return 1/2 * (nsp + nsn)