# Copyright (c) 2026 Jakub Więckowski & contributors

"""Unit tests for pyifdm.methods.validator.Validator"""

from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyifdm.methods.validator import Validator


def test_validate_ifs_element_ok():
    Validator.validate_ifs_element([0.4, 0.3])
    Validator.validate_ifs_element((0.4, 0.3, 0.3))
    Validator.validate_ifs_element(np.array([0.5, 0.5]))


def test_validate_ifs_element_bad_length():
    with pytest.raises(ValueError, match="must have length 2"):
        Validator.validate_ifs_element([0.5])
    with pytest.raises(ValueError, match="must have length 2"):
        Validator.validate_ifs_element([0.1, 0.2, 0.3, 0.4])


def test_validate_ifs_element_bad_values():
    with pytest.raises(ValueError, match="membership"):
        Validator.validate_ifs_element([1.5, 0.0])
    with pytest.raises(ValueError, match="non-membership"):
        Validator.validate_ifs_element([0.2, -0.1])
    with pytest.raises(ValueError, match="μ \\+ ν must be"):
        Validator.validate_ifs_element([0.8, 0.4])


def test_validate_ifs_matrix_ok():
    matrix = np.array([
        [[0.5, 0.3], [0.6, 0.2]],
        [[0.4, 0.4], [0.7, 0.1]],
    ])
    Validator.validate_ifs_matrix(matrix)
    Validator.validate_ifs_matrix(matrix, check_elements=False)


def test_validate_ifs_matrix_bad_shape():
    with pytest.raises(ValueError, match="3-dimensional"):
        Validator.validate_ifs_matrix(np.array([[0.5, 0.3]]))
    with pytest.raises(ValueError, match="last dimension must be 2 or 3"):
        Validator.validate_ifs_matrix(np.zeros((2, 2, 4)))


def test_validate_ifs_matrix_bad_element():
    bad = np.array([[[0.9, 0.2], [0.5, 0.3]]])  # sum > 1
    with pytest.raises(ValueError, match="μ \\+ ν must be"):
        Validator.validate_ifs_matrix(bad)


def test_validate_weights_crisp_ok():
    Validator.validate_weights(np.array([0.2, 0.3, 0.5]))


def test_validate_weights_crisp_bad_sum():
    with pytest.raises(ValueError, match="Sum of crisp weights"):
        Validator.validate_weights(np.array([0.2, 0.3, 0.4]))


def test_validate_weights_ifs_ok():
    w = np.array([[0.3, 0.4], [0.5, 0.2], [0.2, 0.5]])
    Validator.validate_weights(w)


def test_validate_weights_ifs_bad():
    w = np.array([[0.8, 0.4], [0.5, 0.2]])  # first element invalid
    with pytest.raises(ValueError):
        Validator.validate_weights(w)


def test_validate_types_ok():
    Validator.validate_types(np.array([1, -1, 1]))
    Validator.validate_types(np.array([1, 1, 1]), allow_uniform=True)


def test_validate_types_uniform_forbidden():
    with pytest.raises(ValueError, match="not be all identical"):
        Validator.validate_types(np.array([1, 1, 1]))


def test_validate_types_invalid_value():
    with pytest.raises(ValueError, match="only 1.*or -1"):
        Validator.validate_types(np.array([1, 0, -1]))


def test_ifs_validation_full_ok():
    matrix = np.array([
        [[0.5, 0.3], [0.6, 0.2], [0.4, 0.4]],
        [[0.7, 0.1], [0.3, 0.5], [0.55, 0.25]],
    ])
    weights = np.array([0.3, 0.3, 0.4])
    types = np.array([1, -1, 1])
    Validator.ifs_validation(matrix, weights, types, mixed_types=True)


def test_ifs_validation_mismatched_sizes():
    matrix = np.zeros((2, 3, 2))
    matrix[..., 0] = 0.4
    matrix[..., 1] = 0.3
    weights = np.array([0.5, 0.5])  # only 2 weights for 3 criteria
    types = np.array([1, -1, 1])
    with pytest.raises(ValueError, match="Number of criteria"):
        Validator.ifs_validation(matrix, weights, types)
