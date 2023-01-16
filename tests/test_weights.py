# Copyright (c) 2022 Jakub Więckowski

import numpy as np
from pyifdm.weights import *
from pyifdm.methods import ifCOPRAS

def test_burillo_entropy_weights():
    """
        Test veryfing correctness of the standard deviation weights methods.
        Formula: Thakur, P., Kizielewicz, B., Gandotra, N., Shekhovtsov, A., Saini, N., Saeid, A. B., & Sałabun, W. (2021). A New Entropy Measurement for the Analysis of Uncertain Data in MCDA Problems Using Intuitionistic Fuzzy Sets and COPRAS Method. Axioms, 10(4), 335.
        Reference value: Thakur, P., Kizielewicz, B., Gandotra, N., Shekhovtsov, A., Saini, N., Saeid, A. B., & Sałabun, W. (2021). A New Entropy Measurement for the Analysis of Uncertain Data in MCDA Problems Using Intuitionistic Fuzzy Sets and COPRAS Method. Axioms, 10(4), 335.
    """

    matrix = np.array([
        [[0.41, 0.38], [0.48, 0.57], [0.36, 0.43], [0.33, 0.37], [0.28, 0.34]],
        [[0.46, 0.37], [0.48, 0.39], [0.37, 0.41], [0.35, 0.44], [0.51, 0.39]],
        [[0.36, 0.39], [0.21, 0.37], [0.41, 0.38], [0.28, 0.34], [0.32, 0.46]],
        [[0.51, 0.39], [0.37, 0.45], [0.32, 0.46], [0.37, 0.57], [0.37, 0.45]]
    ])

    calculated_weights = burillo_entropy_weights(matrix)
    types = np.array([1, -1, 1, -1, 1])
    if_copras = ifCOPRAS()
    results = if_copras(matrix, calculated_weights, types)

    reference_results = np.array([1.000, 1.037, 1.295, 1.051])

    assert all(np.round(results, 3) == reference_results)

def test_entropy_ifs_weights():
    """
        Test veryfing correctness of the standard deviation weights methods.
        Formula: Ying-Yu, W., & De-Jian, Y. (2011, September). Extended VIKOR for multi-criteria decision making problems under intuitionistic environment. In 2011 International Conference on Management Science & Engineering 18th Annual Conference Proceedings (pp. 118-122). IEEE.
        Reference value: Ying-Yu, W., & De-Jian, Y. (2011, September). Extended VIKOR for multi-criteria decision making problems under intuitionistic environment. In 2011 International Conference on Management Science & Engineering 18th Annual Conference Proceedings (pp. 118-122). IEEE.
    """

    matrix = np.array([
        [[0.75, 0.1], [0.6, 0.25], [0.8, 0.2]],
        [[0.8, 0.15], [0.68, 0.2], [0.45, 0.5]],
        [[0.4, 0.45], [0.75, 0.05], [0.6, 0.3]]
    ])
    calculated_weights = entropy_weights(matrix)
    reference_weights = np.array([0.2452, 0.1826, 0.5722])

    assert (np.round(calculated_weights, 4) == reference_weights).all()

def test_liu_entropy_weights():
    """
        Test veryfing correctness of the standard deviation weights methods.
        Formula: Liu, M., & Ren, H. (2014). A new intuitionistic fuzzy entropy and application in multi-attribute decision making. Information, 5(4), 587-601.
        Reference value: Liu, M., & Ren, H. (2014). A new intuitionistic fuzzy entropy and application in multi-attribute decision making. Information, 5(4), 587-601.
    """

    matrix = np.array([
        [[0.4, 0.1]]
    ])

    calculated_weights = liu_entropy_weights(matrix)
    reference_results = np.array([0.7883])

    assert all(np.round(calculated_weights, 4) == reference_results)

def test_szmidt_entropy_weights():
    """
        Test veryfing correctness of the standard deviation weights methods.
        Formula: Szmidt, E., & Kacprzyk, J. (2001). Entropy for intuitionistic fuzzy sets. Fuzzy sets and systems, 118(3), 467-477.
        Reference value: Szmidt, E., & Kacprzyk, J. (2001). Entropy for intuitionistic fuzzy sets. Fuzzy sets and systems, 118(3), 467-477.
    """

    matrix = np.array([
        [[1/2, 0, 1/2]],
        [[0, 1/2, 1/2]]
    ])

    calculated_weights = szmidt_entropy_weights(matrix)
    reference_results = np.array([1/2])

    assert all(np.round(calculated_weights, 3) == reference_results)

def test_thakur_entropy_weights():
    """
        Test veryfing correctness of the standard deviation weights methods.
        Formula: Thakur, P., Kizielewicz, B., Gandotra, N., Shekhovtsov, A., Saini, N., Saeid, A. B., & Sałabun, W. (2021). A New Entropy Measurement for the Analysis of Uncertain Data in MCDA Problems Using Intuitionistic Fuzzy Sets and COPRAS Method. Axioms, 10(4), 335.
        Reference value: Thakur, P., Kizielewicz, B., Gandotra, N., Shekhovtsov, A., Saini, N., Saeid, A. B., & Sałabun, W. (2021). A New Entropy Measurement for the Analysis of Uncertain Data in MCDA Problems Using Intuitionistic Fuzzy Sets and COPRAS Method. Axioms, 10(4), 335.
    """
    matrix = np.array([
        [[0.41, 0.38], [0.48, 0.57], [0.36, 0.43], [0.33, 0.37], [0.28, 0.34]],
        [[0.46, 0.37], [0.48, 0.39], [0.37, 0.41], [0.35, 0.44], [0.51, 0.39]],
        [[0.36, 0.39], [0.21, 0.37], [0.41, 0.38], [0.28, 0.34], [0.32, 0.46]],
        [[0.51, 0.39], [0.37, 0.45], [0.32, 0.46], [0.37, 0.57], [0.37, 0.45]]
    ])


    calculated_weights = thakur_entropy_weights(matrix)

    types = np.array([1, -1, 1, -1, 1])
    if_copras = ifCOPRAS()
    results = if_copras(matrix, calculated_weights, types)

    reference_results = np.array([1.000, 1.096, 1.320, 1.086])

    assert all(np.round(results, 3) == reference_results)

def test_ye_entropy_weights():
    """
        Test veryfing correctness of the standard deviation weights methods.
        Formula: Ye, J. (2010). Two effective measures of intuitionistic fuzzy entropy. Computing, 87(1), 55-62.
        Reference value: Ye, J. (2010). Two effective measures of intuitionistic fuzzy entropy. Computing, 87(1), 55-62.
    """

    matrix = np.array([
        [[0.2, 0.5]]
    ])

    calculated_weights = ye_entropy_weights(matrix)
    reference_results = np.array([0.9057])

    assert all(np.round(calculated_weights, 4) == reference_results)

