# Copyright (c) 2026 contributors – unit tests for improved IFS class

"""Unit tests for pyifdm.IFS.ifs.IFS"""

from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyifdm.IFS.ifs import IFS, ifs


def test_valid_construction():
    a = IFS(0.6, 0.2)
    assert a.membership == pytest.approx(0.6)
    assert a.non_membership == pytest.approx(0.2)
    assert a.uncertainty == pytest.approx(0.2)
    assert a.mu == a.membership and a.nu == a.non_membership and a.pi == a.uncertainty


def test_construction_with_explicit_pi():
    a = IFS(0.5, 0.3, 0.2)
    assert a.uncertainty == pytest.approx(0.2)


def test_aliases():
    a = ifs(0.4, 0.4)
    assert isinstance(a, IFS)
    assert a.score() == pytest.approx(0.0)
    assert a.accuracy() == pytest.approx(0.8)


def test_invalid_membership_range():
    with pytest.raises(ValueError, match="membership must be in"):
        IFS(1.5, 0.0)
    with pytest.raises(ValueError, match="membership must be in"):
        IFS(-0.1, 0.2)


def test_invalid_non_membership_range():
    with pytest.raises(ValueError, match="non_membership must be in"):
        IFS(0.3, 1.2)
    with pytest.raises(ValueError, match="non_membership must be in"):
        IFS(0.3, -0.01)


def test_invalid_sum():
    with pytest.raises(ValueError, match="must be ≤ 1"):
        IFS(0.7, 0.5)


def test_boundary_values():
    assert IFS(0.0, 0.0).uncertainty == pytest.approx(1.0)
    assert IFS(1.0, 0.0).uncertainty == pytest.approx(0.0)
    assert IFS(0.0, 1.0).uncertainty == pytest.approx(0.0)
    assert IFS(0.5, 0.5).uncertainty == pytest.approx(0.0)


def test_from_array_len2():
    a = IFS.from_array([0.55, 0.25])
    assert a.membership == pytest.approx(0.55)
    assert a.non_membership == pytest.approx(0.25)
    assert a.uncertainty == pytest.approx(0.20)


def test_from_array_len3():
    a = IFS.from_array([0.4, 0.3, 0.3])
    assert a.uncertainty == pytest.approx(0.3)


def test_from_array_invalid_length():
    with pytest.raises(ValueError, match="length 2 or 3"):
        IFS.from_array([0.5])


def test_to_array():
    a = IFS(0.6, 0.2)
    assert np.allclose(a.to_array(True), [0.6, 0.2, 0.2])
    assert np.allclose(a.to_array(False), [0.6, 0.2])


def test_intersection_union_complement():
    a = IFS(0.6, 0.2)
    b = IFS(0.4, 0.3)
    inter = a & b
    union = a | b
    assert inter.membership == pytest.approx(0.4)
    assert inter.non_membership == pytest.approx(0.3)
    assert union.membership == pytest.approx(0.6)
    assert union.non_membership == pytest.approx(0.2)
    comp = ~a
    assert comp.membership == pytest.approx(0.2)
    assert comp.non_membership == pytest.approx(0.6)


def test_addition_multiplication():
    a = IFS(0.5, 0.3)
    b = IFS(0.4, 0.2)
    s = a + b
    # μ = 0.5+0.4-0.5*0.4 = 0.7, ν = 0.3*0.2 = 0.06
    assert s.membership == pytest.approx(0.7)
    assert s.non_membership == pytest.approx(0.06)
    p = a * b
    assert p.membership == pytest.approx(0.2)
    assert p.non_membership == pytest.approx(0.3 + 0.2 - 0.3 * 0.2)


def test_power():
    a = IFS(0.64, 0.2)
    p = a ** 2
    assert p.membership == pytest.approx(0.4096)
    with pytest.raises(ValueError, match="Power must be greater"):
        _ = a ** 0


def test_equality_and_hash():
    a = IFS(0.5, 0.3)
    b = IFS(0.5, 0.3)
    c = IFS(0.5, 0.2)
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    assert len({a, b, c}) == 2


def test_dominance():
    a = IFS(0.7, 0.1)
    b = IFS(0.5, 0.3)
    assert a.dominance(b) is True
    assert b.dominance(a) is False


def test_score_accuracy():
    a = IFS(0.7, 0.1)
    assert a.score() == pytest.approx(0.6)
    assert a.accuracy() == pytest.approx(0.8)


def test_jaccard_similarity():
    a = IFS(0.6, 0.2)
    b = IFS(0.6, 0.2)
    assert a.similarity_jaccard(b) == pytest.approx(1.0)
    c = IFS(0.0, 1.0)
    # non-zero similarity expected
    assert 0.0 <= a.similarity_jaccard(c) <= 1.0


def test_owa_aggregation():
    a = IFS(0.6, 0.2)  # ν=0.2, π=0.2, μ=0.6
    w = [0.2, 0.3, 0.5]
    result = a.owa_aggregation(w)
    expected = 0.2 * 0.2 + 0.3 * 0.2 + 0.5 * 0.6
    assert result == pytest.approx(expected)
    with pytest.raises(ValueError):
        a.owa_aggregation([0.5, 0.5])  # wrong length
    with pytest.raises(ValueError):
        a.owa_aggregation([0.5, 0.5, 0.5])  # sum != 1


def test_repr_str():
    a = IFS(0.6, 0.2)
    assert "0.6" in repr(a)
    assert "Membership" in str(a)
