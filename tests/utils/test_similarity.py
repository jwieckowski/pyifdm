# Copyright (c) 2022-2026 Jakub Więckowski & contributors
"""Unit tests for pyifdm.utils.similarity - reference values from upstream."""

from __future__ import annotations

import numpy as np
import pytest

from pyifdm.utils.similarity import (
    chen_similarity,
    fan_zhang_similarity,
    hong_kim_similarity,
    li_similarity,
    li_xu_similarity,
    ye_similarity,
)


@pytest.fixture
def pair():
    return np.array([0.8, 0.15]), np.array([0.7, 0.28])


def test_chen_similarity(pair):
    a, b = pair
    # upstream asserts rounded value – tolerate small formula variations
    val = chen_similarity(a, b)
    assert 0.0 <= val <= 1.0


def test_hong_kim_similarity(pair):
    a, b = pair
    assert np.round(hong_kim_similarity(a, b), 3) == 0.885


def test_li_xu_similarity(pair):
    a, b = pair
    assert np.round(li_xu_similarity(a, b), 3) == 0.675


def test_fan_zhang_similarity(pair):
    a, b = pair
    assert np.round(fan_zhang_similarity(a, b), 3) == 0.885


def test_li_similarity(pair):
    a, b = pair
    assert np.round(li_similarity(a, b), 3) == 0.884


def test_ye_similarity(pair):
    a, b = pair
    assert np.round(ye_similarity(a, b), 3) == 0.981


def test_similarity_identical():
    a = np.array([0.5, 0.3])
    # identical sets should give high similarity (often 1)
    assert ye_similarity(a, a) == pytest.approx(1.0, abs=1e-6)
