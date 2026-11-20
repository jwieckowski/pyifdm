# Copyright (c) 2022-2026 Jakub Więckowski & contributors

"""Unit tests for pyifdm.helpers"""

from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyifdm.helpers import rank, generate_ifs_matrix


def test_rank_descending_no_ties():
    prefs = np.array([0.1, 0.5, 0.3, 0.9])
    # higher is better: 0.9 → 1, 0.5 → 2, 0.3 → 3, 0.1 → 4
    r = rank(prefs, descending=True)
    assert np.allclose(r, [4, 2, 3, 1])


def test_rank_ascending():
    prefs = np.array([0.1, 0.5, 0.3, 0.9])
    # lower is better: 0.1 → 1, 0.3 → 2, 0.5 → 3, 0.9 → 4
    r = rank(prefs, descending=False)
    assert np.allclose(r, [1, 3, 2, 4])


def test_rank_with_ties():
    prefs = np.array([0.0, 2.0, 3.0, 2.0])
    r = rank(prefs, descending=True)
    # 3 → 1, two 2's → 2.5, 0 → 4
    assert np.allclose(r, [4.0, 2.5, 1.0, 2.5])


def test_rank_all_equal():
    prefs = np.array([0.5, 0.5, 0.5])
    r = rank(prefs)
    assert np.allclose(r, [2.0, 2.0, 2.0])


def test_generate_ifs_matrix_shape_and_bounds():
    m, n = 4, 5
    matrix = generate_ifs_matrix(m, n)
    assert matrix.shape == (m, n, 2)
    assert np.all(matrix >= 0)
    assert np.all(matrix <= 1)
    # μ + ν ≤ 1 for every element
    assert np.all(matrix[:, :, 0] + matrix[:, :, 1] <= 1.0 + 1e-9)


def test_generate_ifs_matrix_reproducible_with_seed():
    np.random.seed(42)
    a = generate_ifs_matrix(2, 2)
    np.random.seed(42)
    b = generate_ifs_matrix(2, 2)
    assert np.allclose(a, b)
