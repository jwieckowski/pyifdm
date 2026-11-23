# Copyright (c) 2022-2026 Jakub Więckowski & contributors

from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from .base import IFMCDAMethod
from pyifdm.utils.distance import hamming_distance

class ifVIKOR(IFMCDAMethod):
    """
    Intuitionistic Fuzzy VIKOR. Returns preference rows (S, R, Q); lower is better.
    """

    descending = False
    requires_mixed_types = False

    def __init__(
        self,
        distance: Callable = hamming_distance,
        normalization: Optional[Callable] = None,
        v: float = 0.5
    ) -> None:
        super().__init__()
        self.distance = distance
        self.normalization = normalization
        self.v = float(v)

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

        # postive and negative ideal solution
        pis, nis = np.zeros((nmatrix.shape[1], ), dtype=object), np.zeros((nmatrix.shape[1], ), dtype=object)
        for j in range(matrix.shape[1]):
            pis[j] = nmatrix[np.argmax(nmatrix[:, j, 0]), j]
            nis[j] = nmatrix[np.argmin(nmatrix[:, j, 0]), j]

        f = 1
        if 'normalized' in self.distance.__name__:
            f = 1/(2*matrix.shape[1])

        # validate data
        if any([all(p == n) for p, n in zip(pis, nis)]):
            raise ValueError('Matrix should not contain same values within a single column')

        # calculation of S and R rankings
        S, R, Q = np.zeros((nmatrix.shape[0], )), np.zeros((nmatrix.shape[0], )), np.zeros((nmatrix.shape[0], ))
        for i in range(matrix.shape[0]):
            if self.distance.__name__== 'normalized_euclidean_distance':
                S[i] = np.sum([weights[j] * (np.sqrt(f * self.distance(pis[j], nmatrix[i, j])) / np.sqrt(f * self.distance(pis[j], nis[j]))) for j in range(nmatrix.shape[1])])
                R[i] = np.max([weights[j] * (np.sqrt(f * self.distance(pis[j], nmatrix[i, j])) / np.sqrt(f * self.distance(pis[j], nis[j]))) for j in range(nmatrix.shape[1])])
            elif self.distance.__name__== 'normalized_hamming_distance':
                S[i] = np.sum([weights[j] * ((f * self.distance(pis[j], nmatrix[i, j])) / (f * self.distance(pis[j], nis[j]))) for j in range(nmatrix.shape[1])])
                R[i] = np.max([weights[j] * ((f * self.distance(pis[j], nmatrix[i, j])) / (f * self.distance(pis[j], nis[j]))) for j in range(nmatrix.shape[1])])
            else:
                S[i] = np.sum([weights[j] * (self.distance(pis[j], nmatrix[i, j]) / self.distance(pis[j], nis[j]))for j in range(nmatrix.shape[1])])
                R[i] = np.max([weights[j] * (self.distance(pis[j], nmatrix[i, j]) / self.distance(pis[j], nis[j]))for j in range(nmatrix.shape[1])])

        # calculation of the compromise ranking Q
        Q = self.v * ((S - np.min(S)) / (np.max(S) - np.min(S))) + (1 - self.v) * ((R - np.min(R)) / (np.max(R) - np.min(R)))

        return np.nan_to_num(S), np.nan_to_num(R), np.nan_to_num(Q)
