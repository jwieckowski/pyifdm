# Copyright (c) 2022-2026 Jakub Więckowski & contributors
"""Unit tests for pyifdm.utils.normalization."""

from __future__ import annotations

import numpy as np
import pytest

from pyifdm.utils.normalization import (
    ecer_normalization,
    max_normalization,
    minmax_normalization,
    supriya_normalization,
    swap_normalization,
)


def _ifs_matrix():
    # 3 alternatives × 2 criteria, (μ, ν)
    return np.array(
        [
            [[0.6, 0.2], [0.4, 0.3]],
            [[0.5, 0.4], [0.7, 0.1]],
            [[0.8, 0.1], [0.3, 0.5]],
        ],
        dtype=float,
    )


def test_swap_normalization_shape_and_bounds():
    matrix = _ifs_matrix()
    types = np.array([1, -1])
    result = swap_normalization(matrix, types)
    assert result.shape == matrix.shape
    assert np.all(result >= 0)
    assert np.all(result <= 1 + 1e-9)


def test_minmax_normalization_shape():
    matrix = _ifs_matrix()
    types = np.array([1, -1])
    result = minmax_normalization(matrix, types)
    assert result.shape == matrix.shape


def test_max_normalization_shape():
    matrix = _ifs_matrix()
    types = np.array([1, -1])
    result = max_normalization(matrix, types)
    assert result.shape == matrix.shape


def test_supriya_normalization_shape():
    matrix = _ifs_matrix()
    result = supriya_normalization(matrix)
    assert result.shape == matrix.shape


def test_ecer_normalization_classic_matrix():
    # 2-D numeric matrix from upstream (not full IFS triples)
    matrix = np.array(
        [
            [0.677, 0.660, 0.612, 0.668, 0.564, 0.610, 0.677],
            [0.363, 0.466, 0.436, 0.451, 0.801, 0.337, 0.255],
            [0.697, 0.638, 0.586, 0.683, 0.711, 0.672, 0.644],
            [0.795, 0.815, 0.847, 0.754, 0.618, 1.000, 0.710],
            [1.000, 1.000, 1.000, 1.000, 0.801, 1.000, 1.000],
            [0.474, 0.590, 0.600, 0.536, 0.627, 0.710, 0.754],
            [0.311, 0.420, 0.288, 0.473, 1.000, 0.292, 0.683],
            [0.837, 0.852, 1.000, 0.837, 0.261, 0.863, 1.000],
            [0.396, 0.363, 0.451, 0.638, 0.400, 0.392, 0.697],
            [0.767, 0.643, 0.628, 0.741, 0.840, 0.754, 0.734],
        ]
    )
    types = np.array([1, 1, 1, 1, 1, -1, -1])
    reference = np.array(
        [
            [0.677, 0.660, 0.612, 0.668, 0.564, 0.479, 0.377],
            [0.363, 0.466, 0.436, 0.451, 0.801, 0.866, 1.000],
            [0.697, 0.638, 0.586, 0.683, 0.711, 0.435, 0.396],
            [0.795, 0.815, 0.847, 0.754, 0.618, 0.292, 0.359],
            [1.000, 1.000, 1.000, 1.000, 0.801, 0.292, 0.255],
            [0.474, 0.590, 0.600, 0.536, 0.627, 0.411, 0.338],
            [0.311, 0.420, 0.288, 0.473, 1.000, 1.000, 0.373],
            [0.837, 0.852, 1.000, 0.837, 0.261, 0.338, 0.255],
            [0.396, 0.363, 0.451, 0.638, 0.400, 0.745, 0.366],
            [0.767, 0.643, 0.628, 0.741, 0.840, 0.387, 0.347],
        ]
    )
    calculated = ecer_normalization(matrix, types)
    assert np.allclose(np.round(calculated, 3), np.round(reference, 3))
