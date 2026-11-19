# Copyright (c) 2022-2026 Jakub Więckowski & contributors
"""Unit tests for pyifdm.utils.distance - reference values from upstream."""

from __future__ import annotations

import numpy as np
import pytest

from pyifdm.utils.distance import (
    euclidean_distance,
    grzegorzewski_distance,
    hamming_distance,
    hausdorf_euclidean_distance,
    luo_distance,
    normalized_euclidean_distance,
    normalized_hamming_distance,
    wang_xin_distance_1,
    wang_xin_distance_2,
    yang_chiclana_distance,
)


def test_euclidean_distance():
    x = np.array([0.25, 0.25])
    y = np.array([0.5, 0.5])
    assert np.round(euclidean_distance(x, y), 4) == np.round(np.sqrt(3) / 4, 4)


def test_grzegorzewski_distance():
    x = np.array([0.5, 0.5])
    y = np.array([0.25, 0.25])
    assert grzegorzewski_distance(x, y) == 0.25


def test_hamming_distance():
    x = np.array([0.25, 0.25])
    y = np.array([0.5, 0.5])
    assert hamming_distance(x, y) == 0.5


def test_hausdorf_euclidean_distance():
    x = np.array([0.8, 0.2])
    y = np.array([0.6, 0.3])
    assert np.round(hausdorf_euclidean_distance(x, y), 3) == 0.04


def test_luo_distance():
    x = np.array([0.4, 0.6])
    y = np.array([0.3, 0.5])
    assert np.round(luo_distance(x, y), 4) == 0.0167


def test_normalized_euclidean_distance_aggregate():
    x = np.array([[0.25, 0.25], [0.2, 0.6], [0.3, 0.2]])
    y = np.array([[0.5, 0.5], [0.3, 0.2], [0.5, 0.2]])
    value = np.sqrt(
        1 / (2 * 3) * np.sum(
            [normalized_euclidean_distance(xx, yy) for xx, yy in zip(x, y)]
        )
    )
    assert np.round(value, 3) == 0.345


def test_normalized_hamming_distance_aggregate():
    x = np.array([[0.25, 0.25], [0.2, 0.6], [0.3, 0.2]])
    y = np.array([[0.5, 0.5], [0.3, 0.2], [0.5, 0.2]])
    value = 1 / (2 * 3) * np.sum(
        [normalized_hamming_distance(xx, yy) for xx, yy in zip(x, y)]
    )
    assert np.round(value, 3) == 0.367


def test_wang_xin_distance_1_aggregate():
    x = np.array([[0.3, 0.6], [0.5, 0.4], [0.7, 0.1]])
    y = np.array([[0.4, 0.6], [0.6, 0.3], [0.5, 0.2]])
    value = 1 / 3 * np.sum(
        [wang_xin_distance_1(xx, yy) for xx, yy in zip(x, y)]
    )
    assert np.round(value, 3) == 0.117


def test_wang_xin_distance_2_aggregate():
    x = np.array([[0.3, 0.6], [0.5, 0.4], [0.7, 0.1]])
    y = np.array([[0.4, 0.6], [0.6, 0.3], [0.5, 0.2]])
    value = 1 / 3 * np.sum(
        [wang_xin_distance_2(xx, yy) for xx, yy in zip(x, y)]
    )
    assert np.round(value, 3) == 0.100


def test_yang_chiclana_distance():
    x = np.array([0.4, 0.6])
    y = np.array([0.3, 0.5])
    assert np.round(yang_chiclana_distance(x, y), 3) == 0.100


def test_distance_symmetry():
    a = np.array([0.6, 0.2])
    b = np.array([0.4, 0.3])
    assert euclidean_distance(a, b) == pytest.approx(euclidean_distance(b, a))
    assert hamming_distance(a, b) == pytest.approx(hamming_distance(b, a))
