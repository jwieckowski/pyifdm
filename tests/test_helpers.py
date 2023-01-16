# Copyright (c) 2022 Jakub Więckowski

import numpy as np
from pyifdm.helpers import *


def test_rank():
    """
        Test veryfing correctness of the rank method.
        Reference value: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html
    """
    preferences = np.array([0, 2, 3, 2])
    calculated_rank = rank(preferences)
    reference_rank = np.array([1, 2.5, 4, 2.5])
    assert (rank(calculated_rank) == reference_rank).all()


def test_generate_ifs_matrix():
    """
        Test veryfing correctness of the random generate Intuitionistic Fuzzy matrix method
    """

    matrix = generate_ifs_matrix(3, 3)

    assert matrix.shape[0] == 3
    assert matrix.shape[1] == 3
    assert matrix.shape[2] == 2
    assert np.min(matrix) >= 0
    assert np.max(matrix) <= 1
