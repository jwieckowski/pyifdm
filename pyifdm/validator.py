# Copyright (c) 2022-2026 Jakub Więckowski & contributors

from __future__ import annotations

from typing import Optional, Union
import numpy as np


class Validator:
    """
    Central validation helpers for Intuitionistic Fuzzy decision matrices,
    weights and criteria types.
    """

    # ------------------------------------------------------------------
    # Low-level IFS element checks
    # ------------------------------------------------------------------
    @staticmethod
    def validate_ifs_element(
        element: Union[list, tuple, np.ndarray],
        *,
        tol: float = 1e-9,
        name: str = "IFS element",
    ) -> None:
        """
        Validate a single IFS value given as length-2 (μ, v) or length-3 (μ, v, π).

        Raises
        ------
        ValueError
            If the element violates IFS axioms.
        """
        arr = np.asarray(element, dtype=float).ravel()
        if arr.size not in (2, 3):
            raise ValueError(
                f"{name} must have length 2 (μ, v) or 3 (μ, v, π), got length {arr.size}"
            )

        mu, nu = float(arr[0]), float(arr[1])
        if not (0.0 - tol <= mu <= 1.0 + tol):
            raise ValueError(f"{name}: membership μ must be in [0, 1], got {mu}")
        if not (0.0 - tol <= nu <= 1.0 + tol):
            raise ValueError(f"{name}: non-membership v must be in [0, 1], got {nu}")
        if mu + nu > 1.0 + tol:
            raise ValueError(
                f"{name}: μ + v must be ≤ 1, got {mu} + {nu} = {mu + nu}"
            )

        if arr.size == 3:
            pi = float(arr[2])
            if not (0.0 - tol <= pi <= 1.0 + tol):
                raise ValueError(
                    f"{name}: uncertainty π must be in [0, 1], got {pi}"
                )
            # soft check – many sources tolerate small floating-point drift
            if abs(mu + nu + pi - 1.0) > 1e-5:
                # we do not raise, only guarantee that the core (μ, v) is valid
                pass

    # ------------------------------------------------------------------
    # Matrix / weights / types
    # ------------------------------------------------------------------
    @staticmethod
    def validate_input(
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
    ) -> None:
        """
        Check that the number of criteria matches the length of weights and types.
        """
        n_criteria = matrix.shape[1]

        if weights.ndim == 1:
            if weights.shape[0] != n_criteria or types.shape[0] != n_criteria:
                raise ValueError(
                    f"Number of criteria ({n_criteria}) must equal "
                    f"number of weights ({weights.shape[0]}) and types ({types.shape[0]})"
                )
        elif weights.ndim == 2:
            # IFS weights: shape (n_criteria, 2) or (n_criteria, 3)
            if weights.shape[0] != n_criteria or types.shape[0] != n_criteria:
                raise ValueError(
                    f"Number of criteria ({n_criteria}) must equal "
                    f"number of weight rows ({weights.shape[0]}) and types ({types.shape[0]})"
                )
        else:
            raise ValueError(
                f"weights must be 1-D (crisp) or 2-D (IFS), got ndim={weights.ndim}"
            )

    @staticmethod
    def validate_ifs_matrix(
        matrix: np.ndarray,
        *,
        tol: float = 1e-9,
        check_elements: bool = True,
    ) -> None:
        """
        Validate the decision matrix shape and (optionally) every IFS element.
        """
        if matrix.ndim != 3:
            raise ValueError(
                f"IFS matrix must be 3-dimensional (alternatives × criteria × components), "
                f"got ndim={matrix.ndim}"
            )
        if matrix.shape[2] not in (2, 3):
            raise ValueError(
                f"IFS matrix last dimension must be 2 or 3, got {matrix.shape[2]}"
            )

        if check_elements:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    Validator.validate_ifs_element(
                        matrix[i, j],
                        tol=tol,
                        name=f"matrix[{i},{j}]",
                    )

    @staticmethod
    def validate_weights(
        weights: np.ndarray,
        *,
        tol: float = 1e-3,
        require_crisp_sum: bool = True,
    ) -> None:
        """
        Validate crisp or IFS weights.

        For crisp weights the sum must be approximately 1.
        For IFS weights every element must be a valid IFS.
        """
        weights = np.asarray(weights)

        if weights.ndim == 1:
            # crisp
            if require_crisp_sum and abs(np.sum(weights) - 1.0) > tol:
                raise ValueError(
                    f"Sum of crisp weights should equal 1 (within {tol}), "
                    f"got {np.sum(weights)}"
                )
            if np.any(weights < -tol) or np.any(weights > 1.0 + tol):
                raise ValueError("Crisp weights must lie in [0, 1]")
        elif weights.ndim == 2:
            # IFS weights
            if weights.shape[1] not in (2, 3):
                raise ValueError(
                    f"IFS weights must have 2 or 3 columns, got {weights.shape[1]}"
                )
            for i, w in enumerate(weights):
                Validator.validate_ifs_element(
                    w, tol=1e-9, name=f"weights[{i}]"
                )
        else:
            raise ValueError(
                f"weights must be 1-D or 2-D, got ndim={weights.ndim}"
            )

    @staticmethod
    def validate_types(types: np.ndarray, *, allow_uniform: bool = False) -> None:
        """
        Validate criteria types (1 = profit, -1 = cost).

        By default mixed types are required (as in the original package).
        Set allow_uniform=True to permit all-profit or all-cost problems.
        """
        types = np.asarray(types)
        unique = np.unique(types)
        if not np.all(np.isin(unique, [-1, 1])):
            raise ValueError(
                f"Criteria types must be only 1 (profit) or -1 (cost), got {unique}"
            )
        if not allow_uniform and len(unique) == 1:
            raise ValueError(
                "Criteria types should not be all identical "
                "(use allow_uniform=True if intentional)"
            )

    # ------------------------------------------------------------------
    # High-level convenience
    # ------------------------------------------------------------------
    @staticmethod
    def ifs_validation(
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
        *,
        mixed_types: bool = False,
        check_elements: bool = True,
        tol: float = 1e-9,
    ) -> None:
        """
        Run the full validation suite used by IF-MCDA methods.

        Parameters
        ----------
        matrix : ndarray
            Decision matrix (m × n × 2/3).
        weights : ndarray
            Crisp (n,) or IFS (n × 2/3) weights.
        types : ndarray
            Criteria types (±1).
        mixed_types : bool, default False
            If True, require that both profit and cost criteria are present.
        check_elements : bool, default True
            If True, validate every IFS triple/pair in the matrix.
        tol : float
            Numerical tolerance for range checks.
        """
        Validator.validate_input(matrix, weights, types)
        Validator.validate_ifs_matrix(
            matrix, tol=tol, check_elements=check_elements
        )
        Validator.validate_weights(weights)
        if mixed_types:
            Validator.validate_types(types, allow_uniform=False)
