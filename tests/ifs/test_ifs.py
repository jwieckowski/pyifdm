# Copyright (c) 2023 Jakub Więckowski

import numpy as np
from pyifdm.ifs import IFS

def test_equality():
    """
        Test verifying correctness of Intuitionistic Fuzzy Sets equality check.
        Reference value: Self-calculated empirical verification
    """

    ifs1 = IFS(0.8, 0.1)
    ifs2 = IFS(0.8, 0.1)
    assert ifs1 == ifs2

def test_ifs_add():
    """
        Test verifying correctness of Intuitionistic Fuzzy Sets addition check.
        Reference value: Self-calculated empirical verification
    """

    ifs1 = IFS(0.6, 0.2)
    ifs2 = IFS(0.8, 0.1)
    result = ifs1 + ifs2
    assert result.membership == 0.6 + 0.8 - 0.6 * 0.8
    assert result.non_membership == 0.2 - 0.1
    assert result.uncertainty == 1 - (result.membership + result.non_membership)

def test_ifs_sub():
    """
        Test verifying correctness of Intuitionistic Fuzzy Sets subtraction check.
        Reference value: Self-calculated empirical verification
    """

    ifs1 = IFS(0.6, 0.2)
    ifs2 = IFS(0.4, 0.1)
    result = ifs1 - ifs2
    assert result.membership == (0.6 - 0.4) / (1 - 0.4)
    assert result.non_membership == 0.2 / 0.1
    assert result.uncertainty == 1 - (result.membership + result.non_membership)

def test_ifs_mul():
    """
        Test verifying correctness of Intuitionistic Fuzzy Sets multiplication check.
        Reference value: Self-calculated empirical verification
    """

    ifs1 = IFS(0.6, 0.2)
    ifs2 = IFS(0.8, 0.1)
    result = ifs1 * ifs2
    assert result.membership == 0.6 * 0.8
    assert result.non_membership == 0.2 + 0.1 - 0.2 * 0.1
    assert result.uncertainty == 1 - (result.membership + result.non_membership)

def test_ifs_div():
    """
        Test verifying correctness of Intuitionistic Fuzzy Sets division check.
        Reference value: Self-calculated empirical verification
    """

    ifs1 = IFS(0.6, 0.2)
    ifs2 = IFS(0.4, 0.1)
    result = ifs1 / ifs2
    assert result.membership == 0.6 / 0.4
    assert result.non_membership == (0.2 - 0.1) / (1 - 0.1)
    assert result.uncertainty == 1 - (result.membership + result.non_membership)

def test_ifs_pow():
    """
        Test verifying correctness of Intuitionistic Fuzzy Sets intersection operations.
        Reference value: Self-calculated empirical verification
    """

    ifs = IFS(0.6, 0.2)
    result = ifs ** 2
    assert result.membership == 0.6 ** 2
    assert result.non_membership == 1 - (1 - 0.2) ** 2
    assert result.uncertainty == 1 - (result.membership + result.non_membership)


def test_intersection():
    """
        Test verifying correctness of Intuitionistic Fuzzy Sets intersection operations.
        Reference value: Self-calculated empirical verification
    """

    ifs1 = IFS(0.6, 0.2)
    ifs2 = IFS(0.8, 0.1)
    result = ifs1 & ifs2
    assert result.membership == min(0.6, 0.8)
    assert result.non_membership == max(0.2, 0.1)
    assert result.uncertainty == 1 - (result.membership + result.non_membership)


def test_union():
    """
        Test verifying correctness of Intuitionistic Fuzzy Sets union operations.
        Reference value: Self-calculated empirical verification
    """

    ifs1 = IFS(0.6, 0.2)
    ifs2 = IFS(0.8, 0.1)
    result = ifs1 | ifs2
    assert result.membership == max(0.6, 0.8)
    assert result.non_membership == min(0.2, 0.1)
    assert result.uncertainty == 1 - (result.membership + result.non_membership)


def test_complement():
    """
        Test verifying correctness of Intuitionistic Fuzzy Sets complement operations.
        Reference value: Self-calculated empirical verification
    """

    ifs1 = IFS(0.6, 0.2)
    result = ~ifs1
    assert result == IFS(0.4, 0.8)

def test_owa_aggregation():
    """
        Test verifying correctness of Intuitionistic Fuzzy Sets OWA aggregation.
        Reference value: Self-calculated empirical verification
    """

    ifs = IFS(0.6, 0.2, 0.1)
    weights = [0.3, 0.4, 0.3]
    result = ifs.owa_aggregation(weights)
    expected_result = np.dot(weights, [0.2, 0.1, 0.6])
    assert result == expected_result

def test_dominance():
    """
        Test verifying correctness of Intuitionistic Fuzzy Sets dominance check.
        Reference value: Self-calculated empirical verification
    """

    ifs1 = IFS(0.6, 0.2)
    ifs2 = IFS(0.5, 0.1)
    assert ifs1.dominance(ifs2)

def test_similarity_jaccard():
    """
        Test verifying correctness of Intuitionistic Fuzzy Sets jaccard similarity.
        Reference value: Self-calculated empirical verification
    """

    ifs1 = IFS(0.6, 0.2)
    ifs2 = IFS(0.8, 0.1)
    result = ifs1.similarity_jaccard(ifs2)
    assert np.isclose(result, 0.5714285714285714)

def test_fuzzy_relation():
    """
        Test verifying correctness of Intuitionistic Fuzzy Sets fuzzy relation.
        Reference value: Self-calculated empirical verification
    """

    ifs1 = IFS(0.6, 0.2)
    ifs2 = IFS(0.8, 0.1)
    result = ifs1.fuzzy_relation(ifs2)
    expected_result = np.array([[0.6, 0.1, 0.1], [0.2, 0.1, 0.1], [0.2, 0.0, 0.0]])
    assert np.allclose(result, expected_result)

def test_aggregate_multiple():
    """
        Test verifying correctness of Intuitionistic Fuzzy Sets aggregation of multiple IFS.
        Reference value: Self-calculated empirical verification
    """

    ifs1 = IFS(0.6, 0.2)
    ifs2 = IFS(0.8, 0.1)
    ifs3 = IFS(0.4, 0.3)
    ifs4 = IFS(0.7, 0.2)
    ifs_list = [ifs1, ifs2, ifs3, ifs4]
    weights_list = [0.2, 0.3, 0.2, 0.3]
    result = IFS.aggregate_multiple(ifs_list, weights_list)
    expected_result = IFS(0.66, 0.17)
    assert result == expected_result
