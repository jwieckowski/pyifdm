# Copyright (c) 2023 Jakub Więckowski

import numpy as np
from pyifdm.IFS import IFS

def test_init():
    ifs = IFS(0.6, 0.2)
    assert ifs.membership == 0.6
    assert ifs.non_membership == 0.2
    assert ifs.uncertainty == 0.2

def test_repr():
    ifs = IFS(0.6, 0.2, 0.2)
    assert repr(ifs) == "IFS(0.6, 0.2, 0.2)"

def test_str():
    ifs = IFS(0.6, 0.2, 0.2)
    assert str(ifs) == "Membership: 0.6, Non-membership: 0.2, Uncertainty: 0.2"

def test_eq():
    ifs1 = IFS(0.6, 0.2, 0.2)
    ifs2 = IFS(0.6, 0.2, 0.2)
    assert ifs1 == ifs2

def test_add():
    ifs1 = IFS(0.6, 0.2, 0.2)
    ifs2 = IFS(0.8, 0.1, 0.05)
    result = ifs1 + ifs2
    assert np.round(result.membership, 2) == 0.92
    assert np.round(result.non_membership, 2) == 0.10
    assert np.round(result.uncertainty, 2) == 0.00

def test_sub():
    ifs1 = IFS(0.6, 0.2, 0.2)
    ifs2 = IFS(0.4, 0.2, 0.2)
    result = ifs1 - ifs2
    assert np.round(result.membership, 2) == 0.00
    assert np.round(result.non_membership, 2) == 1.00
    assert np.round(result.uncertainty, 2) == 0.00

def test_mul():
    ifs1 = IFS(0.6, 0.2, 0.2)
    ifs2 = IFS(0.8, 0.1, 0.1)
    result = ifs1 * ifs2
    assert result == IFS(0.48, 0.28, 0.24)

def test_truediv():
    ifs1 = IFS(0.6, 0.2, 0.2)
    ifs2 = IFS(0.8, 0.1, 0.1)
    result = ifs1 / ifs2
    assert np.round(result.membership, 2) == 0.75
    assert np.round(result.non_membership, 2) == 0.11
    assert np.round(result.uncertainty, 2) == 0.14

def test_pow():
    ifs = IFS(0.6, 0.2, 0.2)
    result = ifs ** 2
    assert np.round(result.membership, 2) == 0.36
    assert np.round(result.non_membership, 2) == 0.36
    assert np.round(result.uncertainty, 2) == 0.28

def test_and():
    ifs1 = IFS(0.6, 0.2, 0.2)
    ifs2 = IFS(0.8, 0.1, 0.1)
    result = ifs1 & ifs2
    assert np.round(result.membership, 2) == 0.60
    assert np.round(result.non_membership, 2) == 0.20
    assert np.round(result.uncertainty, 2) == 0.20

def test_or():
    ifs1 = IFS(0.6, 0.2, 0.2)
    ifs2 = IFS(0.8, 0.1, 0.1)
    result = ifs1 | ifs2
    assert np.round(result.membership, 2) == 0.80
    assert np.round(result.non_membership, 2) == 0.10
    assert np.round(result.uncertainty, 2) == 0.10

def test_invert():
    ifs = IFS(0.6, 0.2, 0.2)
    result = ~ifs
    assert result == IFS(0.2, 0.6, 0.2)

def test_dominance():
    ifs1 = IFS(0.6, 0.2, 0.2)
    ifs2 = IFS(0.8, 0.1, 0.1)
    assert ifs1.dominance(ifs2) is False

def test_owa_aggregation():
    ifs = IFS(0.6, 0.2, 0.2)
    weights = [0.3, 0.4, 0.3]
    result = ifs.owa_aggregation(weights)
    assert np.isclose(result, 0.32)

def test_similarity_jaccard():
    ifs1 = IFS(0.6, 0.2, 0.2)
    ifs2 = IFS(0.8, 0.1, 0.1)
    result = ifs1.similarity_jaccard(ifs2)
    assert np.round(result, 4) == 0.6667

def test_fuzzy_relation():
    ifs1 = IFS(0.6, 0.2, 0.2)
    ifs2 = IFS(0.8, 0.1, 0.1)
    result = ifs1.fuzzy_relation(ifs2)

    assert all(result[0] == [0.6, 0.1, 0.1])
