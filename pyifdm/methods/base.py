# Copyright (c) 2022-2026 Jakub Więckowski & contributors
# Unified base class for Intuitionistic Fuzzy MCDA methods

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np

from pyifdm.helpers import rank
from pyifdm.step_logger import StepLogger
from pyifdm.validator import Validator


ArrayLike = Union[np.ndarray, list]


class IFMCDAMethod(ABC):
    """
    Abstract base class for Intuitionistic Fuzzy MCDA methods.

    Provides:

    - execution wrapper via ``__call__``,
    - shared IFS validation,
    - intermediate step storage (``intermediate_results``),
    - optional ``StepLogger`` integration,
    - ranking of preferences.

    Subclasses implement ``_evaluate`` with the method-specific algorithm.

    Attributes
    ----------
    preferences :
        Final preference values from the last evaluation.
    intermediate_results :
        Named intermediate steps recorded during the last evaluation.
    logger :
        Optional step logger bound for the current run.
    descending :
        If True, higher preference means a better alternative.
    requires_mixed_types :
        If True, validation requires both profit and cost criteria.
    """

    descending: bool = True
    requires_mixed_types: bool = False

    def __init__(self, logger: Optional[StepLogger] = None) -> None:
        self.logger: Optional[StepLogger] = logger
        self.preferences: Optional[np.ndarray] = None
        self.intermediate_results: dict[str, dict[str, Any]] = {}

    def __call__(
        self,
        matrix: ArrayLike,
        weights: ArrayLike,
        types: ArrayLike,
        *args: Any,
        **kwargs: Any,
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
        self.preferences = None
        self.intermediate_results.clear()

        matrix_arr = np.asarray(matrix, dtype=float)
        weights_arr = np.asarray(weights, dtype=float)
        types_arr = np.asarray(types)

        self._validate_input(matrix_arr, weights_arr, types_arr, *args, **kwargs)
        self.logger = self._prepare_logger(self.logger)
        if self.logger is not None:
            self.logger.clear()

        try:
            preferences = self._evaluate(matrix_arr, weights_arr, types_arr, *args, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"'{self.method_name}' failed during calculation: {exc}") from exc

        try:
            preferences = np.asarray(preferences, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"'{self.method_name}._evaluate' must return numeric array-like data.") from exc

        if self.logger is not None:
            try:
                if "console" in self.logger.output:
                    self.logger._write_console()
                if "json" in self.logger.output:
                    self.logger.save()
            except Exception as exc:
                raise RuntimeError(f"Logger failed for '{self.method_name}': {exc}") from exc

        self.preferences = preferences
        return preferences

    def rank(self) -> np.ndarray:
        """
        Rank alternatives based on the last computed preferences.

        Returns
        -------
        ndarray
            Ranking (1 = best). For multi-row preferences (e.g. VIKOR)
            a 2-D array is returned.
        """
        if self.preferences is None:
            raise AttributeError(
                f"{self.method_name}: call the method first before requesting a ranking."
            )
        try:
            prefs = self.preferences
            if prefs.ndim == 2 and prefs.shape[0] <= 5:
                return np.array(
                    [rank(row, self.descending) for row in prefs]
                )
            return rank(prefs, self.descending)
        except Exception as exc:
            raise ValueError(f"Error occurred in ranking calculation: {exc}") from exc

    @abstractmethod
    def _evaluate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Core algorithm. Must return preference values (1-D or 2-D).

        Validation, storage of preferences, and logging lifecycle are
        handled by the base class. Use ``_log_step`` inside this method
        to record intermediate results.
        """
        pass


    def _validate_input(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Common IFS input validation. Override to extend.
        """
        Validator.ifs_validation(
            matrix,
            weights,
            types,
            mixed_types=self.requires_mixed_types,
            check_elements=True,
        )

    def _prepare_logger(
        self, logger: Optional[StepLogger]
    ) -> Optional[StepLogger]:
        if logger is None:
            return None
        if not isinstance(logger, StepLogger):
            raise TypeError(f"'logger' must be a StepLogger instance or None, got {type(logger).__name__}."
            )
        try:
            logger._set_method(self.method_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to bind logger to method '{self.method_name}': {exc}") from exc
        return logger

    def _store_step(
        self,
        name: str,
        data: Any,
        description: str = "",
    ) -> None:
        """
        Store an intermediate computation result in ``intermediate_results``.
        """
        try:
            if isinstance(data, np.ndarray):
                entry: dict[str, Any] = {
                    "shape": list(data.shape),
                    "data": data.tolist(),
                }
            else:
                entry = {"data": data}
        except Exception as exc:
            raise ValueError(
                f"Failed to store intermediate step '{name}' for "
                f"'{self.method_name}': {exc}"
            ) from exc

        if description:
            entry["description"] = description
        self.intermediate_results[name] = entry

    def _log_step(
        self,
        name: str,
        data: Any,
        description: str = "",
        logger: Optional[StepLogger] = None,
    ) -> None:
        """
        Record a step into ``intermediate_results`` and optionally forward
        it to the active logger.

        Parameters
        ----------
        name :
            Step identifier (e.g. ``"normalized_matrix"``).
        data :
            Intermediate value to store.
        description :
            Optional human-readable explanation.
        logger :
            Logger to use. Defaults to ``self.logger``.
        """
        self._store_step(name, data, description)
        active = self.logger if logger is None else logger
        if active is not None:
            try:
                active.log(name, data, description)
            except Exception as exc:
                raise RuntimeError(f"Logger failed while recording step '{name}' for '{self.method_name}': {exc}") from exc

    @property
    def method_name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        def _repr_value(value: Any) -> str:
            try:
                if isinstance(value, np.ndarray):
                    return f"array(shape={value.shape})"
                if callable(value):
                    return getattr(value, "__name__", repr(value))
                return repr(value)
            except Exception:
                return f"<unrepresentable {type(value).__name__}>"

        skip = {"preferences", "intermediate_results", "logger"}
        params = ", ".join(
            f"{key}={_repr_value(val)}"
            for key, val in self.__dict__.items()
            if not key.startswith("_") and key not in skip
        )
        return f"{self.method_name}({params})"
