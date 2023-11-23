# Copyright (c) 2022-2023 Jakub Więckowski

import numpy as np
from pyifdm.methods.ifs.distance import *


def test_euclidean_distance():
    """
        Test veryfing correctness of the Euclidean distance formula.
        Formula: Çalı, S., & Balaman, Ş. Y. (2019). A novel outranking based multi criteria group decision making methodology integrating ELECTRE and VIKOR under intuitionistic fuzzy environment. Expert Systems with Applications, 119, 36-50.
        Reference value: Szmidt, E., & Kacprzyk, J. (2000). Distances between intuitionistic fuzzy sets. Fuzzy sets and systems, 114(3), 505-518.
    """
    x = np.array([0.25, 0.25])
    y = np.array([0.5, 0.5])
    calculated_value = euclidean_distance(x, y)
    reference_value = np.sqrt(3) / 4

    assert np.round(calculated_value, 4) == np.round(reference_value, 4)

def test_grzegorzewski_distance():
    """
        Test veryfing correctness of the Grzegorzewski distance formula.
        Formula: Grzegorzewski, P. (2004). Distances between intuitionistic fuzzy sets and/or interval-valued fuzzy sets based on the Hausdorff metric. Fuzzy sets and systems, 148(2), 319-328.
        Reference value: Grzegorzewski, P. (2004). Distances between intuitionistic fuzzy sets and/or interval-valued fuzzy sets based on the Hausdorff metric. Fuzzy sets and systems, 148(2), 319-328.
    """
    x = np.array([0.5, 0.5])
    y = np.array([0.25, 0.25])
    calculated_value = grzegorzewski_distance(x, y)
    reference_value = 0.25

    assert calculated_value == reference_value

def test_hamming_distance():
    """
        Test veryfing correctness of the Hamming distance formula.
        Formula: Çalı, S., & Balaman, Ş. Y. (2019). A novel outranking based multi criteria group decision making methodology integrating ELECTRE and VIKOR under intuitionistic fuzzy environment. Expert Systems with Applications, 119, 36-50.
        Reference value: Self-calculated empirical verification
    """
    x = np.array([0.25, 0.25])
    y = np.array([0.5, 0.5])
    calculated_value = hamming_distance(x, y)
    reference_value = 0.5

    assert calculated_value == reference_value

def test_hausdorf_euclidean_distance():
    """
        Test veryfing correctness of the Hausdorf measure-based Euclidean distance formula.
        Formula: Li, M., Wu, C., Zhang, L., & You, L. N. (2015). An intuitionistic fuzzy-TODIM method to solve distributor evaluation and selection problem. International Journal of Simulation Modelling, 14(3), 511-524.
        Reference value: Self-calculated empirical verification
    """
    x = np.array([0.8, 0.2])
    y = np.array([0.6, 0.3])
    calculated_value = hausdorf_euclidean_distance(x, y)
    reference_value = 0.04

    assert np.round(calculated_value, 3) == reference_value

def test_luo_distance():
    """
        Test veryfing correctness of the Luo distance formula.
        Formula: Li, Y. (2021). IF-MABAC Method for Evaluating the Intelligent Transportation System with Intuitionistic Fuzzy Information. Journal of Mathematics, 2021.
        Reference value: Self-calculated empirical verification
    """
    x = np.array([0.4, 0.6])
    y = np.array([0.3, 0.5])
    calculated_value = luo_distance(x, y)
    reference_value = 0.0167

    assert np.round(calculated_value, 4) == reference_value

def test_normalized_euclidean_distance():
    """
        Test veryfing correctness of the normalized Euclidean distance formula.
        Formula: Çalı, S., & Balaman, Ş. Y. (2019). A novel outranking based multi criteria group decision making methodology integrating ELECTRE and VIKOR under intuitionistic fuzzy environment. Expert Systems with Applications, 119, 36-50.
        Reference value: Self-calculated empirical verification
    """
    x = np.array([[0.25, 0.25], [0.2, 0.6], [0.3, 0.2]])
    y = np.array([[0.5, 0.5], [0.3, 0.2], [0.5, 0.2]])
    calculated_value = np.sqrt(1/(2*3) * np.sum([normalized_euclidean_distance(xx, yy) for xx, yy in zip(x, y)]))
    reference_value = 0.345

    assert np.round(calculated_value, 3) == reference_value

def test_normalized_hamming_distance():
    """
        Test veryfing correctness of the normalized Hamming distance formula.
        Formula: Çalı, S., & Balaman, Ş. Y. (2019). A novel outranking based multi criteria group decision making methodology integrating ELECTRE and VIKOR under intuitionistic fuzzy environment. Expert Systems with Applications, 119, 36-50.
        Reference value: Self-calculated empirical verification
    """
    x = np.array([[0.25, 0.25], [0.2, 0.6], [0.3, 0.2]])
    y = np.array([[0.5, 0.5], [0.3, 0.2], [0.5, 0.2]])
    calculated_value = 1/(2*3) * np.sum([normalized_hamming_distance(xx, yy) for xx, yy in zip(x, y)])
    reference_value = 0.367

    assert np.round(calculated_value, 3) == reference_value

def test_wang_xin_distance_1():
    """
        Test veryfing correctness of the Wang Xin distance 1 formula.
        Formula: Wang, W., & Xin, X. (2005). Distance measure between intuitionistic fuzzy sets. Pattern recognition letters, 26(13), 2063-2069.
        Reference value: Wang, W., & Xin, X. (2005). Distance measure between intuitionistic fuzzy sets. Pattern recognition letters, 26(13), 2063-2069.
    """
    x = np.array([[0.3, 0.6], [0.5, 0.4], [0.7, 0.1]])
    y = np.array([[0.4, 0.6], [0.6, 0.3], [0.5, 0.2]])
    calculated_value = 1/3 * np.sum([wang_xin_distance_1(xx, yy) for xx, yy in zip(x, y)])
    reference_value = 0.117

    assert np.round(calculated_value, 3) == reference_value

def test_wang_xin_distance_2():
    """
        Test veryfing correctness of the Wang Xin distance 2 formula.
        Formula: Wang, W., & Xin, X. (2005). Distance measure between intuitionistic fuzzy sets. Pattern recognition letters, 26(13), 2063-2069.
        Reference value: Wang, W., & Xin, X. (2005). Distance measure between intuitionistic fuzzy sets. Pattern recognition letters, 26(13), 2063-2069.
    """
    x = np.array([[0.3, 0.6], [0.5, 0.4], [0.7, 0.1]])
    y = np.array([[0.4, 0.6], [0.6, 0.3], [0.5, 0.2]])
    calculated_value = 1/3 * np.sum([wang_xin_distance_2(xx, yy) for xx, yy in zip(x, y)])
    reference_value = 0.100

    assert np.round(calculated_value, 3) == reference_value

def test_yang_chiclana_distance():
    """
        Test veryfing correctness of the Yang & Chiclana distance formula.
        Formula: Yang, Y., & Chiclana, F. (2012). Consistency of 2D and 3D distances of intuitionistic fuzzy sets. Expert Systems with Applications, 39(10), 8665-8670.
        Reference value: Self-calculated empirical verification
    """
    x = np.array([0.4, 0.6])
    y = np.array([0.3, 0.5])
    calculated_value = yang_chiclana_distance(x, y)
    reference_value = 0.100

    assert np.round(calculated_value, 3) == reference_value
