# Copyright (c) 2023-2026 Jakub Więckowski & contributors

from __future__ import annotations

from typing import Optional, Sequence, Union
import numpy as np


Number = Union[int, float, np.floating]


class IFS:
    """
    Intuitionistic Fuzzy Set (IFS).

    An IFS is defined by membership (μ), non-membership (ν) and hesitation/
    uncertainty (π) degrees satisfying:

        0 ≤ μ, ν ≤ 1
        μ + ν ≤ 1
        π = 1 - μ - ν   (when not supplied explicitly)

    Parameters
    ----------
    membership : float
        Degree of membership μ ∈ [0, 1].
    non_membership : float
        Degree of non-membership ν ∈ [0, 1].
    uncertainty : float, optional
        Hesitation degree π. If omitted it is computed as 1 - μ - ν.
        When provided it must satisfy μ + ν + π ≈ 1 (within tolerance).

    Raises
    ------
    ValueError
        If the values violate the IFS axioms.
    """

    __slots__ = ("_membership", "_non_membership", "_uncertainty")

    def __init__(
        self,
        membership: Number,
        non_membership: Number,
        uncertainty: Optional[Number] = None,
        *,
        tol: float = 1e-9,
    ) -> None:
        mu = float(membership)
        nu = float(non_membership)

        if not (0.0 - tol <= mu <= 1.0 + tol):
            raise ValueError(f"membership must be in [0, 1], got {mu}")
        if not (0.0 - tol <= nu <= 1.0 + tol):
            raise ValueError(f"non_membership must be in [0, 1], got {nu}")
        if mu + nu > 1.0 + tol:
            raise ValueError(
                f"membership + non_membership must be ≤ 1, got {mu} + {nu} = {mu + nu}"
            )

        # clamp tiny numerical noise
        mu = min(max(mu, 0.0), 1.0)
        nu = min(max(nu, 0.0), 1.0)

        if uncertainty is not None:
            pi = float(uncertainty)
            if not (0.0 - tol <= pi <= 1.0 + tol):
                raise ValueError(f"uncertainty must be in [0, 1], got {pi}")
            if abs(mu + nu + pi - 1.0) > 1e-6:
                # soft handling – recompute to keep consistency
                pi = max(0.0, 1.0 - mu - nu)
        else:
            pi = max(0.0, 1.0 - mu - nu)

        self._membership = mu
        self._non_membership = nu
        self._uncertainty = pi

    @property
    def membership(self) -> float:
        return self._membership

    @property
    def non_membership(self) -> float:
        return self._non_membership

    @property
    def uncertainty(self) -> float:
        return self._uncertainty

    # aliases commonly used in literature
    @property
    def mu(self) -> float:
        return self._membership

    @property
    def nu(self) -> float:
        return self._non_membership

    @property
    def pi(self) -> float:
        return self._uncertainty

    @classmethod
    def from_array(
        cls, arr: Sequence[Number], *, tol: float = 1e-9
    ) -> "IFS":
        """
        Create IFS from a sequence of length 2 (μ, ν) or 3 (μ, ν, π).
        """
        arr = list(arr)
        if len(arr) == 2:
            return cls(arr[0], arr[1], tol=tol)
        if len(arr) == 3:
            return cls(arr[0], arr[1], arr[2], tol=tol)
        raise ValueError(
            f"IFS array must have length 2 or 3, got length {len(arr)}"
        )

    def to_array(self, with_uncertainty: bool = True) -> np.ndarray:
        """Return (μ, ν) or (μ, ν, π) as numpy array."""
        if with_uncertainty:
            return np.array(
                [self._membership, self._non_membership, self._uncertainty],
                dtype=float,
            )
        return np.array([self._membership, self._non_membership], dtype=float)

    def __repr__(self) -> str:
        return (
            f"IFS({self._membership:.6g}, {self._non_membership:.6g}, "
            f"{self._uncertainty:.6g})"
        )

    def __str__(self) -> str:
        return (
            f"Membership: {self._membership:.6g}, "
            f"Non-membership: {self._non_membership:.6g}, "
            f"Uncertainty: {self._uncertainty:.6g}"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IFS):
            return NotImplemented
        return (
            abs(self._membership - other._membership) < 1e-9
            and abs(self._non_membership - other._non_membership) < 1e-9
            and abs(self._uncertainty - other._uncertainty) < 1e-9
        )

    def __hash__(self) -> int:
        return hash(
            (
                round(self._membership, 9),
                round(self._non_membership, 9),
                round(self._uncertainty, 9),
            )
        )

    def __add__(self, other: "IFS") -> "IFS":
        if not isinstance(other, IFS):
            return NotImplemented
        mu = self._membership + other._membership - self._membership * other._membership
        nu = self._non_membership * other._non_membership
        return IFS(mu, nu)

    def __sub__(self, other: "IFS") -> "IFS":
        """
        Subtraction based on the condition from the original package.
        Falls back to (0, 1) when the operation is undefined.
        """
        if not isinstance(other, IFS):
            return NotImplemented
        # avoid division by zero
        if other._non_membership == 0 or (1 - other._membership) == 0:
            return IFS(0.0, 1.0)
        if self._non_membership / other._non_membership <= (
            (1 - self._membership) / (1 - other._membership)
        ):
            mu = (self._membership - other._membership) / (1 - other._membership)
            nu = self._non_membership / other._non_membership
            return IFS(max(0.0, mu), min(1.0, nu))
        return IFS(0.0, 1.0)

    def __mul__(self, other: "IFS") -> "IFS":
        if not isinstance(other, IFS):
            return NotImplemented
        mu = self._membership * other._membership
        nu = (
            self._non_membership
            + other._non_membership
            - self._non_membership * other._non_membership
        )
        return IFS(mu, nu)

    def __truediv__(self, other: "IFS") -> "IFS":
        if not isinstance(other, IFS):
            return NotImplemented
        if other._membership == 0 or (1 - other._non_membership) == 0:
            return IFS(1.0, 0.0)
        if self._membership / other._membership <= (
            (1 - self._non_membership) / (1 - other._non_membership)
        ):
            mu = self._membership / other._membership
            nu = (self._non_membership - other._non_membership) / (
                1 - other._non_membership
            )
            return IFS(min(1.0, mu), max(0.0, nu))
        return IFS(1.0, 0.0)

    def __pow__(self, y: Number) -> "IFS":
        y = float(y)
        if y <= 0:
            raise ValueError("Power must be greater than 0")
        mu = self._membership ** y
        nu = 1.0 - (1.0 - self._non_membership) ** y
        return IFS(mu, nu)

    def __and__(self, other: "IFS") -> "IFS":
        """Intersection (min μ, max ν)."""
        if not isinstance(other, IFS):
            return NotImplemented
        mu = min(self._membership, other._membership)
        nu = max(self._non_membership, other._non_membership)
        return IFS(mu, nu)

    def __or__(self, other: "IFS") -> "IFS":
        """Union (max μ, min ν)."""
        if not isinstance(other, IFS):
            return NotImplemented
        mu = max(self._membership, other._membership)
        nu = min(self._non_membership, other._non_membership)
        return IFS(mu, nu)

    def __invert__(self) -> "IFS":
        """Complement: (ν, μ)."""
        return IFS(self._non_membership, self._membership)

    def dominance(self, other: "IFS") -> bool:
        """True if this IFS dominates `other` (higher μ, lower ν and π)."""
        if not isinstance(other, IFS):
            raise TypeError("other must be an IFS instance")
        eps = 1e-12
        return (
            self._membership + eps >= other._membership
            and self._non_membership <= other._non_membership + eps
            and self._uncertainty <= other._uncertainty + eps
        )

    def owa_aggregation(self, weights: Sequence[Number]) -> float:
        """
        Ordered Weighted Averaging on the triple (ν, π, μ).

        `weights` must sum to 1 and have length 3.
        """
        w = np.asarray(weights, dtype=float)
        if w.shape != (3,):
            raise ValueError("OWA weights must be a sequence of length 3")
        if abs(w.sum() - 1.0) > 1e-6:
            raise ValueError("OWA weights must sum to 1")
        values = np.array(
            [self._non_membership, self._uncertainty, self._membership]
        )
        return float(np.dot(w, values))

    def similarity_jaccard(self, other: "IFS") -> float:
        """Jaccard similarity between two IFS."""
        if not isinstance(other, IFS):
            raise TypeError("other must be an IFS instance")
        inter = (
            min(self._membership, other._membership)
            + min(self._non_membership, other._non_membership)
            + min(self._uncertainty, other._uncertainty)
        )
        union = (
            max(self._membership, other._membership)
            + max(self._non_membership, other._non_membership)
            + max(self._uncertainty, other._uncertainty)
        )
        if union == 0:
            return 1.0
        return inter / union

    def fuzzy_relation(self, other: "IFS") -> np.ndarray:
        """3×3 fuzzy relation matrix (component-wise min)."""
        if not isinstance(other, IFS):
            raise TypeError("other must be an IFS instance")
        a = [self._membership, self._non_membership, self._uncertainty]
        b = [other._membership, other._non_membership, other._uncertainty]
        rel = np.zeros((3, 3))
        for i, ai in enumerate(a):
            for j, bj in enumerate(b):
                rel[i, j] = min(ai, bj)
        return rel

    def score(self) -> float:
        """Simple score function S = μ - ν (common in ranking)."""
        return self._membership - self._non_membership

    def accuracy(self) -> float:
        """Accuracy function H = μ + ν."""
        return self._membership + self._non_membership

def ifs(mu: Number, nu: Number, pi: Optional[Number] = None) -> IFS:
    """Shorthand constructor."""
    return IFS(mu, nu, pi)
