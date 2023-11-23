# Copyright (c) 2022-2023 Jakub Więckowski, Bartłomiej Kizielewicz

import numpy as np
from pyifdm.methods import *
from pyifdm.helpers import rank


def test_ifARAS():
    """
        Test verifying correctness of the Intuitionistic Fuzzy ARAS
        Reference value: Raj Mishra, A., Sisodia, G., Raj Pardasani, K., & Sharma, K. (2020). Multi-criteria IT personnel selection on intuitionistic fuzzy information measures and ARAS methodology. Iranian Journal of Fuzzy Systems, 17(4), 55-68.
    """

    matrix = np.array([
        [
            [0.7, 0.2], [0.6, 0.3], [0.6338, 0.2649], [
                0.5712, 0.3277], [0.5885, 0.3064],
            [0.7, 0.2], [0.5, 0.4], [0.6392, 0.2594], [
                0.5, 0.4], [0.6723, 0.2265],
            [0.3376, 0.5621], [0.6367, 0.2617], [
                0.7, 0.2], [0.5, 0.4], [0.5, 0.4]
        ],
        [
            [0.5667, 0.3326], [0.5384, 0.3608], [
                0.7, 0.2], [0.5331, 0.3662], [0.7, 0.2],
            [0.6674, 0.2313], [0.6, 0.3], [0.6338, 0.2649], [
                0.5667, 0.3326], [0.7, 0.2],
            [0.2, 0.7], [0.6723, 0.2265], [0.9, 0.1], [0.5, 0.4], [0.5, 0.4]
        ],
        [
            [0.9, 0.1], [0.6367, 0.2619], [0.3659, 0.5338], [
                0.4105, 0.4850], [0.3376, 0.5621],
            [0.6367, 0.2619], [0.7, 0.2], [0.3659, 0.5338], [
                0.3659, 0.5338], [0.3352, 0.5645],
            [0.7, 0.2], [0.7, 0.2], [0.7, 0.2], [0.7, 0.2], [0.6, 0.3]
        ],
        [
            [0.5, 0.4], [0.6392, 0.2594], [0.5, 0.4], [
                0.5384, 0.3608], [0.2350, 0.6648],
            [0.6392, 0.2594], [0.5, 0.4], [0.7, 0.2], [0.9, 0.1], [0.7, 0.2],
            [0.7, 0.2], [0.6, 0.3], [0.6367, 0.2619], [
                0.6674, 0.2313], [0.5, 0.4]
        ],
        [
            [0.5, 0.4], [0.6338, 0.2649], [0.3324, 0.5673], [
                0.3659, 0.5338], [0.2, 0.7],
            [0.5712, 0.3277], [0.5, 0.4], [0.5, 0.4], [
                0.6338, 0.2649], [0.6392, 0.2594],
            [0.5375, 0.3603], [0.6, 0.3], [0.4702, 0.4274], [
                0.6697, 0.2291], [0.4811, 0.4164]
        ]
    ])

    weights = np.array([0.0298, 0.0048, 0.0632, 0.0186, 0.1133, 0.0069, 0.0273,
                        0.0427, 0.0905, 0.0592, 0.1081, 0.1081, 0.0662, 0.2610, 0.0006])
    types = np.array([1] * matrix.shape[1])

    if_aras = ifARAS()
    results = if_aras(matrix, weights, types)
    reference_results = np.array([0.9438, 0.9550, 0.9514, 0.9623, 0.9377])
    
    assert all(rank(results) == rank(reference_results))
    assert all(if_aras.rank() == rank(reference_results))

def test_ifCODAS():
    """
        Test verifying correctness of the Intuitionistic Fuzzy CODAS
        Reference value: Buyukozkan, G., & Göçer, F. (2019, August). Prioritizing the strategies to enhance smart city logistics by intuitionistic fuzzy CODAS. In 11th Conference of the European Society for Fuzzy Logic and Technology (EUSFLAT 2019) (pp. 805-811). Atlantis Press.
    """

    matrix = np.array([
        [[0.234, 0.685], [1.0, 0], [0.188, 0.741], [
            0.201, 0.669], [0.085, 0.823], [0.26, 0.592]],
        [[0.18, 0.754], [0.241, 0.649], [0.097, 0.846],
            [1.0, 0], [0.15, 0.8], [1.0, 0]],
        [[0.254, 0.599], [0.197, 0.722], [0.187, 0.737], [
            0.131, 0.791], [0.26, 0.642], [0.256, 0.595]],
        [[0.26, 0.592], [0.304, 0.582], [0.142, 0.797], [
            0.04, 0.896], [0.171, 0.732], [0.142, 0.797]],
        [[0.085, 0.823], [0.26, 0.642], [0.171, 0.732],
            [1.0, 0], [1.0, 0], [0.04, 0.896]]
    ])

    weights = np.array([[0.23, 0.709], [0.425, 0.418], [0.31, 0.546], [
                    0.464, 0.355], [0.383, 0.493], [0.204, 0.732]])
    types = np.array([1, -1, 1, -1, 1, 1])

    if_codas = ifCODAS()
    results = if_codas(matrix, weights, types)
    reference_results = np.array([0.564, 2.328, -3.002, -3.571, 3.681])

    assert all(rank(results) == rank(reference_results))
    assert all(if_codas.rank() == rank(reference_results))

def test_ifCOPRAS():
    """
        Test verifying correctness of the Intuitionistic Fuzzy COPRAS
        Reference value: Thakur, P., Kizielewicz, B., Gandotra, N., Shekhovtsov, A., Saini, N., Saeid, A. B., & Sałabun, W. (2021). A New Entropy Measurement for the Analysis of Uncertain Data in MCDA Problems Using Intuitionistic Fuzzy Sets and COPRAS Method. Axioms, 10(4), 335.
    """

    matrix = np.array([
        [[0.41, 0.38], [0.48, 0.57], [0.36, 0.43], [0.33, 0.37], [0.28, 0.34]],
        [[0.46, 0.37], [0.48, 0.39], [0.37, 0.41], [0.35, 0.44], [0.51, 0.39]],
        [[0.36, 0.39], [0.21, 0.37], [0.41, 0.38], [0.28, 0.34], [0.32, 0.46]],
        [[0.51, 0.39], [0.37, 0.45], [0.32, 0.46], [0.37, 0.57], [0.37, 0.45]]
    ])

    weights = np.array([0.2004, 0.2804, 0.1714, 0.1563, 0.1914])
    types = np.array([1, -1, 1, -1, 1])

    if_copras = ifCOPRAS()
    results = if_copras(matrix, weights, types)
    reference_results = np.array([1.000, 1.096, 1.320, 1.086])

    assert all(np.round(results, 3) == reference_results)
    assert all(if_copras.rank() == rank(reference_results))

def test_ifEDAS():
    """
        Test verifying correctness of the Intuitionistic Fuzzy EDAS
        Reference value: Liang, Y. (2020). An EDAS method for multiple attribute group decision-making under intuitionistic fuzzy environment and its application for evaluating green building energy-saving design projects. Symmetry, 12(3), 484.
    """

    matrix = np.array([
        [[0.4745, 0.5255], [0.4752, 0.5248], [0.2981, 0.7019], [0.43743, 0.5627]],
        [[0.5346, 0.4654], [0.5532, 0.4468], [0.63, 0.37], [0.5901, 0.4099]],
        [[0.4324, 0.5676], [0.403, 0.597], [0.4298, 0.5702], [0.4361, 0.5639]],
        [[0.5235, 0.4765], [0.4808, 0.5192], [0.5667, 0.4333], [0.2913, 0.7087]],
        [[0.4168, 0.5832], [0.4923, 0.5077], [0.4732, 0.5268], [0.4477, 0.5523]]
    ])

    weights = np.array([0.1410, 0.2263, 0.3234, 0.3093])
    types = np.array([1, -1, 1, 1])

    if_edas = ifEDAS()
    results = if_edas(matrix, weights, types)
    reference_results = np.array([0.003, 0.881, 0.348, 0.251, 0.363])

    assert all(np.round(results, 3) == reference_results)
    assert all(if_edas.rank() == rank(reference_results))

def test_ifMABAC():
    """
        Test verifying correctness of the Intuitionistic Fuzzy MABAC
        Reference value: Li, Y. (2021). IF-MABAC Method for Evaluating the Intelligent Transportation System with Intuitionistic Fuzzy Information. Journal of Mathematics, 2021.
    """

    matrix = np.array([
        [[0.4874, 0.3382], [0.3130, 0.5747], [0.5342, 0.3512], [0.5187, 0.3641]],
        [[0.7184, 0.2087], [0.2840, 0.6350], [0.6906, 0.2309], [0.6696, 0.2628]],
        [[0.5039, 0.4103], [0.2863, 0.5766], [0.4662, 0.4452], [0.5123, 0.3796]],
        [[0.5575, 0.3284], [0.3331, 0.5524], [0.5265, 0.3718], [0.5219, 0.3727]],
        [[0.4975, 0.4319], [0.3695, 0.5372], [0.4479, 0.4860], [0.6371, 0.1889]]
    ])
    weights = np.array([0.2793, 0.1699, 0.2845, 0.2663])
    types = np.array([1, -1, 1, 1])

    if_mabac = ifMABAC()
    results = if_mabac(matrix, weights, types)
    reference_ranking = np.array([3, 1, 5, 2, 4])

    assert all(rank(results) == reference_ranking)
    assert all(if_mabac.rank() == reference_ranking)

def test_ifMAIRCA():
    """
        Test verifying correctness of the Intuitionistic Fuzzy MAIRCA
        Reference value: Ecer, F. (2022). An extended MAIRCA method using intuitionistic fuzzy sets for coronavirus vaccine selection in the age of COVID-19. Neural Computing and Applications, 34(7), 5603-5623.
    """

    matrix = np.array([
        [[1, 0, 0], [0.579, 0.321, 0.101], [0.178, 0.7, 0.122], [1, 0, 0], [0.553, 0.346,
                                                                            0.101], [0.679, 0.22, 0.101], [0.329, 0.548, 0.123], [0.525, 0.374, 0.101]],
        [[1, 0, 0], [0.553, 0.346, 0.101], [0.363, 0.525, 0.112], [1, 0, 0], [
            0.654, 0.245, 0.101], [0.676, 0.223, 0.101], [0.368, 0.522, 0.110], [0.755, 0.173, 0.072]],
        [[1, 0, 0], [0.679, 0.220, 0.101], [1, 0, 0], [0.731, 0.185, 0.084],
            [1, 0, 0], [0.676, 0.223, 0.101], [1, 0, 0], [0.85, 0.1, 0.05]],
        [[0.755, 0.173, 0.072], [0.819, 0.120, 0.06], [1, 0, 0], [0.7, 0.2, 0.1], [
            0.731, 0.185, 0.084], [0.679, 0.22, 0.101], [1, 0, 0], [0.755, 0.173, 0.072]],
        [[0.755, 0.173, 0.072], [0.788, 0.141, 0.071], [0.788, 0.141, 0.071],
            [1, 0, 0], [1, 0, 0], [0.679, 0.22, 0.1], [1, 0, 0], [1, 0, 0]],
    ])

    weights = np.array([0.1422, 0.1344, 0.1243, 0.1390,
                        0.1344, 0.0970, 0.1232, 0.1053])
    types = np.array([1, 1, 1, 1, 1, -1, -1, -1])

    if_mairca = ifMAIRCA()
    results = if_mairca(matrix, weights, types)
    reference_results = np.array([0.095, 0.078, 0.078, 0.126, 0.103])

    assert all(np.round(results, 3) == reference_results)
    assert all(if_mairca.rank() == [3, 4, 5, 1, 2])

def test_ifMARCOS():
    """
        Test verifying correctness of the Intuitionistic Fuzzy MARCOS
        Reference value: Ecer, F., & Pamucar, D. (2021). MARCOS technique under intuitionistic fuzzy environment for determining the COVID-19 pandemic performance of insurance companies in terms of healthcare services. Applied Soft Computing, 104, 107199.
    """

    matrix = np.array([
        [[0.584, 0.020, 0.396], [0.553, 0.018, 0.429], [0.452, 0.029, 0.518], [0.568, 0.012, 0.420], [0.329, 0.043, 0.627], [0.449, 0.029, 0.522], [0.584, 0.011, 0.404]],
        [[0.285, 0.577, 0.138], [0.407, 0.478, 0.116], [0.372, 0.506, 0.123], [0.379, 0.484, 0.137], [0.775, 0.163, 0.062], [0.250, 0.600, 0.150], [0.157, 0.692, 0.151]],
        [[0.654, 0.245, 0.101], [0.591, 0.305, 0.104], [0.538, 0.361, 0.101], [0.639, 0.259, 0.101], [0.673, 0.241, 0.086], [0.627, 0.270, 0.102], [0.597, 0.300, 0.104]],
        [[0.769, 0.166, 0.065], [0.788, 0.142, 0.071], [0.827, 0.122, 0.051], [0.717, 0.192, 0.091], [0.571, 0.327, 0.102], [1.000, 0.000, 0.000], [0.667, 0.231, 0.101]],
        [[1.000, 0.000, 0.000], [1.000, 0.000, 0.000], [1.000, 0.000, 0.000], [1.000, 0.000, 0.000], [0.775, 0.163, 0.062], [1.000, 0.000, 0.000], [1.000, 0.000, 0.000]],
        [[0.423, 0.477, 0.100], [0.542, 0.357, 0.101], [0.553, 0.346, 0.101], [0.487, 0.413, 0.100], [0.580, 0.319, 0.101], [0.667, 0.231, 0.101], [0.721, 0.200, 0.079]],
        [[0.220, 0.629, 0.150], [0.360, 0.527, 0.113], [0.194, 0.655, 0.151], [0.414, 0.469, 0.117], [1.000, 0.000, 0.000], [0.199, 0.650, 0.151], [0.639, 0.259, 0.102]],
        [[0.815, 0.129, 0.056], [0.832, 0.117, 0.051], [1.000, 0.000, 0.000], [0.814, 0.126, 0.060], [0.178, 0.689, 0.133], [0.844, 0.106, 0.050], [1.000, 0.000, 0.000]],
        [[0.329, 0.548, 0.123], [0.285, 0.577, 0.138], [0.391, 0.493, 0.116], [0.591, 0.305, 0.104], [0.331, 0.541, 0.128], [0.324, 0.551, 0.125], [0.653, 0.244, 0.103]],
        [[0.733, 0.184, 0.083], [0.597, 0.301, 0.102], [0.581, 0.317, 0.102], [0.705, 0.208, 0.087], [0.816, 0.134, 0.051], [0.721, 0.200, 0.079], [0.696, 0.212, 0.092]]
    ])
    weights = np.array([0.117, 0.131, 0.156, 0.136, 0.137, 0.163, 0.160])
    types = np.array([1, 1, 1, 1, 1, -1, -1])

    if_marcos = ifMARCOS()
    if_marcos(matrix, weights, types)
    reference_results = np.array([7, 2, 6, 3, 1, 9, 8, 4, 10, 5])

    assert all(if_marcos.rank() == reference_results)

def test_ifMOORA():
    """
        Test verifying correctness of the Intuitionistic Fuzzy MOORA
        Reference value: Pérez-Domínguez, L., Alvarado-Iniesta, A., Rodríguez-Borbón, I., & Vergara-Villegas, O. (2015). Intuitionistic fuzzy MOORA for supplier selection. Dyna, 82(191), 34-41.
    """
    
    matrix = np.array([
        [[0.2, 0.7, 0.2], [0.3, 0.6, 0.1], [1, 0, 0], [1, 0, 0]],
        [[0.5, 0.5, 0.1], [0.8, 0.1, 0.1], [0.4, 0.5, 0.1], [0.8, 0.1, 0.1]],
        [[0.8, 0.1, 0.1], [1, 0, 0], [0.8, 0.1, 0.1], [0.2, 0.7, 0.2]],
        [[0.6, 0.3, 0.1], [0.2, 0.7, 0.2], [0.1, 0.8, 0.1], [0.3, 0.6, 0.1]],
        [[0.4, 0.5, 0.1], [0.5, 0.5, 0.1], [0.5, 0.4, 0.1], [0.5, 0.5, 0.1]]
    ], dtype=object)

    weights = np.array([
        [0.64, 0.3, 0.06],
        [0.65, 0.3, 0.05],
        [0.35, 0.6, 0.05],
        [0.23, 0.74, 0.03]
    ])
    types = np.array([-1, 1, 1, 1])

    if_moora = ifMOORA()
    results = if_moora(matrix, weights, types)
    reference_ranking = np.array([1, 2, 3, 5, 4])

    assert all(rank(results) == reference_ranking)
    assert all(if_moora.rank() == reference_ranking)

def test_ifOCRA():
    """
        Test verifying correctness of the Intuitionistic Fuzzy OCRA
        Formula: Mishra, A. R., Rani, P., Cavallaro, F., Hezam, I. M., & Lakshmi, J. (2023). An integrated intuitionistic fuzzy closeness coefficient-based OCRA method for sustainable urban transportation options selection. Axioms, 12(2), 144.
        Reference value: Self-calculated empirical verification
    """
    
    matrix = np.array([
        [[0.698, 0.215, 0.087],	[0.830, 0.130, 0.041],	[0.620, 0.276, 0.103],	[0.830, 0.130, 0.041],	[0.645, 0.260, 0.095]],
        [[0.638, 0.270, 0.092],	[0.751, 0.169, 0.080],	[0.574, 0.323, 0.104],	[0.779, 0.169, 0.052],	[0.690, 0.226, 0.084]],
        [[0.315, 0.584, 0.101],	[0.221, 0.699, 0.080],	[0.322, 0.577, 0.101],	[0.298, 0.602, 0.101],	[0.327, 0.572, 0.101]],
        [[0.384, 0.478, 0.139],	[0.286, 0.614, 0.101],	[0.355, 0.543, 0.103],	[0.392, 0.506, 0.102],	[0.295, 0.604, 0.101]],
        [[0.575, 0.321, 0.104],	[0.632, 0.285, 0.083],	[0.614, 0.283, 0.103],	[0.847, 0.123, 0.030],	[0.703, 0.210, 0.087]],
        [[0.816, 0.151, 0.033],	[0.595, 0.312, 0.094],	[0.661, 0.255, 0.084],	[0.798, 0.164, 0.039],	[0.679, 0.237, 0.084]],
        [[0.618, 0.293, 0.088],	[0.621, 0.296, 0.084],	[0.698, 0.230, 0.072],	[0.712, 0.204, 0.084],	[0.689, 0.224, 0.087]],
        [[0.261, 0.658, 0.082],	[0.379, 0.521, 0.100],	[0.367, 0.530, 0.102],	[0.281, 0.634, 0.085],	[0.286, 0.611, 0.104]],
        [[0.330, 0.567, 0.103],	[0.316, 0.581, 0.103],	[0.296, 0.616, 0.088],	[0.236, 0.679, 0.085],	[0.319, 0.579, 0.103]],
        [[0.811, 0.159, 0.030],	[0.809, 0.145, 0.046],	[0.669, 0.249, 0.082],	[0.562, 0.334, 0.104],	[0.654, 0.263, 0.083]],
        [[0.705, 0.212, 0.084],	[0.703, 0.217, 0.080],	[0.801, 0.162, 0.037],	[0.701, 0.209, 0.091],	[0.698, 0.230, 0.072]]
    ])

    weights = np.array([0.0893, 0.0847, 0.0943, 0.0954, 0.0860, 0.0973, 0.0838, 0.0852, 0.1008, 0.0915, 0.0917])
    types = np.array([1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1])


    M = []
    for j in range(matrix.shape[1]):
        M.append([matrix[i, j] for i in range(matrix.shape[0])])

    matrix = np.array(M)

    if_ocra = ifOCRA()
    results = if_ocra(matrix, weights, types)
    reference_ranking = np.array([4, 2, 5, 1, 3])

    assert all(rank(results) == reference_ranking)
    assert all(if_ocra.rank() == reference_ranking)

def test_ifTOPSIS():
    """
        Test verifying correctness of the Intuitionistic Fuzzy TOPSIS
        Reference value: Boran, F. E., Boran, K. U. R. T. U. L. U. Ş., & Menlik, T. (2012). The evaluation of renewable energy technologies for electricity generation in Turkey using intuitionistic fuzzy TOPSIS. Energy Sources, Part B: Economics, Planning, and Policy, 7(1), 81-90.
    """

    matrix = np.array([
        [[0.84, 0.13, 0.03], [0.64, 0.23, 0.13], [0.79, 0.1, 0.11],
            [0.23, 0.68, 0.09], [0.31, 0.58, 0.11]],
        [[0.57, 0.32, 0.11], [0.21, 0.67, 0.12], [0.63, 0.29, 0.08],
            [0.58, 0.29, 0.13], [0.18, 0.77, 0.05]],
        [[0.21, 0.73, 0.06], [0.38, 0.57, 0.05], [0.71, 0.18, 0.11],
            [0.81, 0.1, 0.09], [0.46, 0.49, 0.05]],
        [[0.42, 0.49, 0.09], [0.54, 0.37, 0.09], [0.61, 0.36, 0.03],
            [0.43, 0.41, 0.16], [0.25, 0.58, 0.17]]
    ])

    weights = np.array([[0.5, 0.45], [0.85, 0.1], [
        0.75, 0.1], [0.5, 0.45], [0.75, 0.1]])

    types = np.array([-1, -1, 1, 1, -1])

    if_topsis = ifTOPSIS()
    results = if_topsis(matrix, weights, types)
    reference_results = np.array([0.27, 0.65, 0.66, 0.41])

    assert all(np.round(results, 2) == reference_results)
    assert all(if_topsis.rank() == rank(reference_results))

def test_ifVIKOR():
    """
        Test verifying correctness of the Intuitionistic Fuzzy VIKOR
        Reference value: Ying-Yu, W., & De-Jian, Y. (2011, September). Extended VIKOR for multi-criteria decision making problems under intuitionistic environment. In 2011 International Conference on Management Science & Engineering 18th Annual Conference Proceedings (pp. 118-122). IEEE.
    """

    v = 0.5
    matrix = np.array([
        [[0.6689, 0.2193], [0.8000, 0.1000], [0.3500, 0.5500], [0.7081, 0.1857], [0.5592, 0.3388]],
        [[0.6062, 0.2717], [0.6495, 0.2331], [0.6071, 0.2912], [0.5000, 0.4000], [0.7562, 0.1383]],
        [[0.5542, 0.3438], [0.7308, 0.1568], [0.8355, 0.1223], [0.8465, 0.1077], [0.7128, 0.1808]],
        [[0.6495, 0.2331], [0.6719, 0.2168], [0.6074, 0.2908], [0.5546, 0.3434], [0.5423, 0.3491]]
    ])
    weights = np.array([0.1950, 0.2129, 0.1980, 0.1966, 0.1976])
    types = np.array([1, -1, 1, -1, 1])

    if_vikor = ifVIKOR(v=v)
    results = if_vikor(matrix, weights, types)

    reference_S = np.array([0.4495, 0.5896, 0.3295, 0.6569])
    reference_R = np.array([0.1980, 0.2129, 0.1950, 0.1976])
    reference_Q = np.array([0.2671, 0.8972, 0.0000, 0.5726])

    assert all(rank(results[0], False) == rank(reference_S, False))
    assert all(rank(results[1], False) == rank(reference_R, False))
    assert all(rank(results[2], False) == rank(reference_Q, False))

    ranks = if_vikor.rank()
    assert all(ranks[0] == rank(reference_S, False))
    assert all(ranks[1] == rank(reference_R, False))
    assert all(ranks[2] == rank(reference_Q, False))

def test_ifWPM():
    """
        Test verifying correctness of the Intuitionistic Fuzzy WPM
        Formula: Mishra, Xiong, L., Zhong, S., Liu, S., Zhang, X., & Li, Y. (2020). An approach for resilient-green supplier selection based on WASPAS, BWM, and TOPSIS under intuitionistic fuzzy sets. Mathematical Problems in Engineering, 2020.
        Reference value: Self-calculated empirical verification
    """
    
    matrix = np.array([
        [[1.00, 0.00], [0.60, 0.28], [0.39, 0.49], [0.59, 0.31], [0.80, 0.15], [0.15, 0.70], [0.75, 0.19], [1.00, 0.00], [0.47, 0.43], [0.72, 0.24], [0.44, 0.44], [0.53, 0.35], [0.45, 0.43]],
        [[0.62, 0.25], [0.68, 0.21], [0.65, 0.23], [0.50, 0.40], [1.00, 0.00], [0.65, 0.30], [1.00, 0.00], [0.74, 0.15], [0.77, 0.13], [0.58, 0.31], [0.67, 0.31], [0.65, 0.23], [0.66, 0.21]],
        [[1.00, 0.00], [1.00, 0.00], [0.44, 0.46], [1.00, 0.00], [0.77, 0.12], [0.63, 0.26], [0.52, 0.37], [0.81, 0.17], [0.54, 0.34], [0.30, 0.57], [0.54, 0.36], [0.54, 0.36], [0.33, 0.55]],
        [[0.53, 0.37], [0.80, 0.15], [0.15, 0.74], [0.73, 0.16], [0.47, 0.43], [0.41, 0.47], [1.00, 0.00], [0.75, 0.21], [0.65, 0.24], [0.55, 0.35], [1.00, 0.00], [1.00, 0.00], [0.35, 0.55]]
    ])

    weights = np.array([0.085, 0.030, 0.244, 0.028, 0.063, 0.089, 0.085, 0.089, 0.041, 0.063, 0.112, 0.022, 0.049])
    types = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    if_wpm = ifWPM()
    results = if_wpm(matrix, weights, types)
    reference_ranking = np.array([3, 1, 2, 4])

    assert all(rank(results) == reference_ranking)
    assert all(if_wpm.rank() == reference_ranking)

def test_ifWSM():
    """
        Test verifying correctness of the Intuitionistic Fuzzy WSM
        Formula: Mishra, Xiong, L., Zhong, S., Liu, S., Zhang, X., & Li, Y. (2020). An approach for resilient-green supplier selection based on WASPAS, BWM, and TOPSIS under intuitionistic fuzzy sets. Mathematical Problems in Engineering, 2020.
        Reference value: Self-calculated empirical verification
    """
    
    matrix = np.array([
        [[1.00, 0.00], [0.60, 0.28], [0.39, 0.49], [0.59, 0.31], [0.80, 0.15], [0.15, 0.70], [0.75, 0.19], [1.00, 0.00], [0.47, 0.43], [0.72, 0.24], [0.44, 0.44], [0.53, 0.35], [0.45, 0.43]],
        [[0.62, 0.25], [0.68, 0.21], [0.65, 0.23], [0.50, 0.40], [1.00, 0.00], [0.65, 0.30], [1.00, 0.00], [0.74, 0.15], [0.77, 0.13], [0.58, 0.31], [0.67, 0.31], [0.65, 0.23], [0.66, 0.21]],
        [[1.00, 0.00], [1.00, 0.00], [0.44, 0.46], [1.00, 0.00], [0.77, 0.12], [0.63, 0.26], [0.52, 0.37], [0.81, 0.17], [0.54, 0.34], [0.30, 0.57], [0.54, 0.36], [0.54, 0.36], [0.33, 0.55]],
        [[0.53, 0.37], [0.80, 0.15], [0.15, 0.74], [0.73, 0.16], [0.47, 0.43], [0.41, 0.47], [1.00, 0.00], [0.75, 0.21], [0.65, 0.24], [0.55, 0.35], [1.00, 0.00], [1.00, 0.00], [0.35, 0.55]]
    ])

    weights = np.array([0.085, 0.030, 0.244, 0.028, 0.063, 0.089, 0.085, 0.089, 0.041, 0.063, 0.112, 0.022, 0.049])
    types = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    if_wsm = ifWSM()
    results = if_wsm(matrix, weights, types)
    reference_ranking = np.array([4, 3, 1, 2])

    assert all(rank(results) == reference_ranking)
    assert all(if_wsm.rank() == reference_ranking)

def test_ifWASPAS():
    """
            Test verifying correctness of the Intuitionistic Fuzzy WASPAS
            Reference value: Mishra, A. R., Singh, R. K., & Motwani, D. (2019). Multi-criteria assessment of cellular mobile telephone service providers using intuitionistic fuzzy WASPAS method with similarity measures. Granular Computing, 4, 511-529.
        """

    v = 0.5
    matrix = np.array([
        [(0.6268, 0.2794), (0.5987, 0.3323), (0.7585, 0.1520), (0.2709, 0.6292), (0.4841, 0.4452), (0.1392, 0.8182), (0.6958, 0.2094), (0.5743, 0.3682)],
        [(0.4520, 0.4711), (0.6728, 0.2196), (0.7207, 0.1866), (0.1020, 0.8858), (0.1500, 0.8017), (0.3508, 0.5584), (0.5935, 0.3042), (0.5346, 0.4087)],
        [(0.1050, 0.8538), (0.0774, 0.9226), (0.6874, 0.2157), (0.1970, 0.7652), (0.4596, 0.4505), (0.5343, 0.4121), (0.3393, 0.5668), (0.1498, 0.7599)],
        [(0.2575, 0.6433), (0.1673, 0.7376), (0.7207, 0.1866), (0.4160, 0.5038), (0.4841, 0.4452), (0.1190, 0.8702), (0.8330, 0.1301), (0.1259, 0.8351)],
        [(0.2181, 0.6829), (0.0774, 0.9226), (0.6164, 0.2930), (0.5278, 0.4182), (0.2516, 0.6476), (0.0912, 0.8934), (0.8162, 0.1474), (0.1696, 0.7357)]
    ])
    weights = np.array([0.1471, 0.1618, 0.0882, 0.1280, 0.1105, 0.1132, 0.1132, 0.1379])
    types = np.array([-1, -1, 1, -1, -1, -1, 1, -1])
    norm = lambda vec, types: vec

    if_waspas = ifWASPAS(v=v)
    results = if_waspas(matrix, weights, types)

    reference_Q = np.array([0.5507, 0.4720, 0.2984, 0.3946, 0.3407])

    assert all(rank(results, False) == rank(reference_Q, False))
