# Copyright (c) 2022 Jakub Więckowski

import numpy as np
from pyifdm.methods.ifs.normalization import *

def test_ecer_normalization():
    """
        Test veryfing correctness of the Ecer normalization formula.
        Formula: Ecer, F., & Pamucar, D. (2021). MARCOS technique under intuitionistic fuzzy environment for determining the COVID-19 pandemic performance of insurance companies in terms of healthcare services. Applied Soft Computing, 104, 107199.
        Reference value: Ecer, F., & Pamucar, D. (2021). MARCOS technique under intuitionistic fuzzy environment for determining the COVID-19 pandemic performance of insurance companies in terms of healthcare services. Applied Soft Computing, 104, 107199.
    """
    matrix = np.array([
        [0.677, 0.660, 0.612, 0.668, 0.564, 0.610, 0.677],
        [0.363, 0.466, 0.436, 0.451, 0.801, 0.337, 0.255],
        [0.697, 0.638, 0.586, 0.683, 0.711, 0.672, 0.644],
        [0.795, 0.815, 0.847, 0.754, 0.618, 1.000, 0.710],
        [1.000, 1.000, 1.000, 1.000, 0.801, 1.000, 1.000],
        [0.474, 0.590, 0.600, 0.536, 0.627, 0.710, 0.754],
        [0.311, 0.420, 0.288, 0.473, 1.000, 0.292, 0.683],
        [0.837, 0.852, 1.000, 0.837, 0.261, 0.863, 1.000],
        [0.396, 0.363, 0.451, 0.638, 0.400, 0.392, 0.697],
        [0.767, 0.643, 0.628, 0.741, 0.840, 0.754, 0.734]
    ])
    types = np.array([1, 1, 1, 1, 1, -1, -1])

    reference_matrix = np.array([
        [0.677, 0.660, 0.612, 0.668, 0.564, 0.479, 0.377],
        [0.363, 0.466, 0.436, 0.451, 0.801, 0.866, 1.000],
        [0.697, 0.638, 0.586, 0.683, 0.711, 0.435, 0.396],
        [0.795, 0.815, 0.847, 0.754, 0.618, 0.292, 0.359],
        [1.000, 1.000, 1.000, 1.000, 0.801, 0.292, 0.255],
        [0.474, 0.590, 0.600, 0.536, 0.627, 0.411, 0.338],
        [0.311, 0.420, 0.288, 0.473, 1.000, 1.000, 0.373],
        [0.837, 0.852, 1.000, 0.837, 0.261, 0.338, 0.255],
        [0.396, 0.363, 0.451, 0.638, 0.400, 0.745, 0.366],
        [0.767, 0.643, 0.628, 0.741, 0.840, 0.387, 0.347]
    ])

    calculated_matrix = ecer_normalization(matrix, types)

    assert np.alltrue(np.round(calculated_matrix, 3) == reference_matrix)

def test_minmax_normalization():
    """
        Test veryfing correctness of the Min-Max normalization formula.
        Formula: Ecer, F. (2022). An extended MAIRCA method using intuitionistic fuzzy sets for coronavirus vaccine selection in the age of COVID-19. Neural Computing and Applications, 34(7), 5603-5623.
        Reference value: Ecer, F. (2022). An extended MAIRCA method using intuitionistic fuzzy sets for coronavirus vaccine selection in the age of COVID-19. Neural Computing and Applications, 34(7), 5603-5623.
    """
    matrix = np.array([
        [1, 0.62493982, 0.25379677, 1, 0.60029762, 0.72101469, 0.39547185, 0.57320893],
        [1, 0.60029762, 0.42201097, 1, 0.69730001, 0.71817811, 0.42575805, 0.78438864],
        [1, 0.72101469, 1, 0.76510189, 1, 0.71817811, 1, 0.86880897],
        [0.78438864, 0.84233003, 1, 0.74050031, 0.76510189, 0.72101469, 1, 0.78438864],
        [0.78438864, 0.81543947, 0.81543947, 1, 1, 0.72112109, 1, 1]
    ])
    types = np.array([1, 1, 1, 1, 1, -1, -1, -1])

    reference_matrix = np.array([
        [1, 0.10181366, 0, 1, 0, 0.03615534, 1, 1],
        [1, 0, 0.22542678, 1, 0.24268655, 1, 0.94990109, 0.50519183],
        [1, 0.49876408, 1, 0.09480388, 1, 1, 0, 0.30738935],
        [0, 1, 1, 0, 0.41231746, 0.03615534, 0, 0.50519183],
        [0,0.88889689, 0.75266721, 1, 1, 0, 0, 0]
    ]).astype(float)

    calculated_matrix = minmax_normalization(matrix, types)

    assert np.alltrue(np.round(calculated_matrix, 4)== np.round(reference_matrix, 4))
    
def test_surpriya_normalization():
    """
        Test veryfing correctness of the Supriya normalization formula.
        Formula: Ejegwa, P. A., Akowe, S. O., Otene, P. M., & Ikyule, J. M. (2014). An overview on intuitionistic fuzzy sets. Int. J. Sci. Technol. Res, 3(3), 142-145.
        Reference value: Ejegwa, P. A., Akowe, S. O., Otene, P. M., & Ikyule, J. M. (2014). An overview on intuitionistic fuzzy sets. Int. J. Sci. Technol. Res, 3(3), 142-145.
    """
    matrix = np.array([
        [[0.6, 0.4]],
        [[0.8, 0.2]],
        [[0.7, 0.3]]
    ])

    reference_matrix = np.array([
        [[0.75, 0.25]],
        [[1.0, 0.0]],
        [[0.875, 0.125]]
    ])

    calculated_matrix = supriya_normalization(matrix)

    assert np.allclose(calculated_matrix, reference_matrix)

def test_normalization_swap():
    """
        Test veryfing correctness of the Swap normalization formula.
        Formula: Liang, Y. (2020). An EDAS method for multiple attribute group decision-making under intuitionistic fuzzy environment and its application for evaluating green building energy-saving design projects. Symmetry, 12(3), 484.
        Reference value: Liang, Y. (2020). An EDAS method for multiple attribute group decision-making under intuitionistic fuzzy environment and its application for evaluating green building energy-saving design projects. Symmetry, 12(3), 484.
    """
    matrix = np.array([
        [[0.4745, 0.5255], [0.4752, 0.5248], [0.2981, 0.7019], [0.43743, 0.5627]],
        [[0.5346, 0.4654], [0.5532, 0.4468], [0.63, 0.37], [0.5901, 0.4099]],
        [[0.4324, 0.5676], [0.4030, 0.5970], [0.4298, 0.5702], [0.4361, 0.5639]],
        [[0.5235, 0.4765], [0.4808, 0.5192], [0.5667, 0.4333], [0.2913, 0.7087]],
        [[0.4168, 0.5832], [0.4923, 0.5077], [0.4732, 0.5268], [0.4477, 0.5523]]
    ])
    types = np.array([1, -1, 1, 1])

    reference_matrix = np.array([
        [[0.4745, 0.5255], [0.5248, 0.4752], [0.2981, 0.7019], [0.43743, 0.5627]],
        [[0.5346, 0.4654], [0.4468, 0.5532], [0.63, 0.37], [0.5901, 0.4099]],
        [[0.4324, 0.5676], [0.5970, 0.4030], [0.4298, 0.5702], [0.4361, 0.5639]],
        [[0.5235, 0.4765], [0.5192, 0.4808], [0.5667, 0.4333], [0.2913, 0.7087]],
        [[0.4168, 0.5832], [0.5077, 0.4923], [0.4732, 0.5268], [0.4477, 0.5523]]
    ])

    calculated_matrix = swap_normalization(matrix, types)

    assert np.alltrue(calculated_matrix == reference_matrix)
