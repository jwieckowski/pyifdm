# Copyright (c) 2022-2026 Jakub Więckowski & contributors
# Unified base class for Intuitionistic Fuzzy MCDA methods

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

import numpy as np

from ..helpers import rank
from .validator import Validator


ArrayLike = Union[np.ndarray, list]


class IFMCDAMethod(ABC):
    """
    Abstract base class for all Intuitionistic Fuzzy MCDA methods.

    Subclasses must implement ``_evaluate``.  The public API is:

    * ``__call__(matrix, weights, types)`` → preference vector/matrix
    * ``rank()`` → ranking derived from the last preferences
    * ``preferences`` – stored result of the last evaluation

    Parameters common to most methods can be accepted in the subclass
    ``__init__`` (normalization, score, distance, …).
    """

    descending: bool = True
    requires_mixed_types: bool = False

    def __init__(self) -> None:
        self.preferences: Optional[np.ndarray] = None

    def __call__(
        self,
        matrix: ArrayLike,
        weights: ArrayLike,
        types: ArrayLike,
    ) -> np.ndarray:
        """
        Evaluate alternatives and store preferences.

        Parameters
        ----------
        matrix :
            Decision matrix of shape (m, n, 2) or (m, n, 3).
        weights :
            Crisp weights (n,) or IFS weights (n, 2/3).
        types :
            Criteria types: 1 = profit, -1 = cost.

        Returns
        -------
        ndarray
            Preference values for the alternatives.
        """
        matrix = np.asarray(matrix, dtype=float)
        weights = np.asarray(weights, dtype=float)
        types = np.asarray(types)

        Validator.ifs_validation(
            matrix,
            weights,
            types,
            mixed_types=self.requires_mixed_types,
            check_elements=True,
        )

        self.preferences = np.asarray(
            self._evaluate(matrix, weights, types), dtype=float
        )
        return self.preferences

    def rank(self) -> np.ndarray:
        """
        Rank alternatives based on the last computed preferences.

        Returns
        -------
        ndarray
            Ranking (1 = best).  For methods that return multiple
            preference vectors (e.g. VIKOR) a 2-D array is returned.
        """
        if self.preferences is None:
            raise AttributeError(
                "Cannot calculate ranking before assessment. "
                "Call the method with (matrix, weights, types) first."
            )
        try:
            prefs = self.preferences
            # VIKOR-style: multiple preference rows
            if prefs.ndim == 2 and prefs.shape[0] <= 5:
                return np.array(
                    [rank(row, self.descending) for row in prefs]
                )
            return rank(prefs, self.descending)
        except Exception as exc:
            raise ValueError(
                f"Error occurred in ranking calculation: {exc}"
            ) from exc

    @abstractmethod
    def _evaluate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
    ) -> np.ndarray:
        """
        Core algorithm.  Must return preference values (1-D or 2-D).
        Validation and storage are handled by the base class.
        """
        ...

    def __repr__(self) -> str:
        params = []
        for attr in ("normalization", "score", "distance", "v"):
            if hasattr(self, attr):
                val = getattr(self, attr)
                name = getattr(val, "__name__", repr(val))
                params.append(f"{attr}={name}")
        return f"{self.__class__.__name__}({', '.join(params)})"
