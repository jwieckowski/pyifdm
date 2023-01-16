# Copyright (c) 2022 Jakub Więckowski

import numpy as np
from pyifdm.methods.ifs.score import *

def test_liu_wang_score():
    """
        Test veryfing correctness of the Liu Wang score formula.
        Formula: Liang, Y. (2020). An EDAS method for multiple attribute group decision-making under intuitionistic fuzzy environment and its application for evaluating green building energy-saving design projects. Symmetry, 12(3), 484. 
        Reference value: Self-calculated empirical verification
    """

    x = np.array([0.6, 0.3])
    calculated_value = liu_wang_score(x)
    reference_value = 0.66

    assert np.round(calculated_value, 2) == reference_value

def test_wei_score():
    """
        Test veryfing correctness of the Wei score formula.
        Formula: Wei, C. P., Gao, Z. H., & Guo, T. T. (2012). An intuitionistic fuzzy entropy measure based on trigonometric function. Control and Decision, 27(4), 571-574.
        Reference value: Self-calculated empirical verification
    """

    x = np.array([0.6, 0.3])
    calculated_value = wei_score(x)
    reference_value = 0.910

    assert np.round(calculated_value, 3) == reference_value

def test_wan_dong_score_1():
    """
        Test veryfing correctness of the Wan Dong score 1 formula.
        Formula: Raj Mishra, A., Sisodia, G., Raj Pardasani, K., & Sharma, K. (2020). Multi-criteria IT personnel selection on intuitionistic fuzzy information measures and ARAS methodology. Iranian Journal of Fuzzy Systems, 17(4), 55-68.
        Reference value: Self-calculated empirical verification   
    """

    x = np.array([0.6, 0.3])
    calculated_value = wan_dong_score_1(x)
    reference_value = 0.575

    assert np.round(calculated_value, 3) == reference_value

def test_wan_dong_score_2():
    """
        Test veryfing correctness of the Wan Dong score 2 formula.
        Formula: Ziquan, X., Jiaqi, Y., Naseem, M. H., Zuquan, X., & Xueheng, L. (2021). Supplier selection of shipbuilding enterprises based on intuitionistic fuzzy multicriteria decision. Mathematical Problems in Engineering, 2021.
        Reference value: Ziquan, X., Jiaqi, Y., Naseem, M. H., Zuquan, X., & Xueheng, L. (2021). Supplier selection of shipbuilding enterprises based on intuitionistic fuzzy multicriteria decision. Mathematical Problems in Engineering, 2021.
    """

    x = np.array([0.5422, 0.3416])
    calculated_value = wan_dong_score_2(x)
    reference_value = 0.6003

    assert np.round(calculated_value, 4) == reference_value

def test_zhang_xu_score_1():
    """
        Test veryfing correctness of the Zhang Xu score 1 formula.
        Formula: Zhang, X., & Xu, Z. (2012). A new method for ranking intuitionistic fuzzy values and its application in multi-attribute decision making. Fuzzy Optimization and Decision Making, 11(2), 135-146. 
        Reference value: Zhang, X., & Xu, Z. (2012). A new method for ranking intuitionistic fuzzy values and its application in multi-attribute decision making. Fuzzy Optimization and Decision Making, 11(2), 135-146.
    """

    x = np.array([0.7209, 0.1551])
    calculated_value = zhang_xu_score_1(x)
    reference_value = 0.7517

    print(calculated_value)

    assert np.round(calculated_value, 4) == reference_value

def test_chen_score_1():
    """
        Test veryfing correctness of the Chen score 1 formula.
        Formula: Kizielewicz, B., Paradowski, B., Więckowski, J., & Sałabun, W. (2022, September). Towards the identification of MARCOS models based on intuitionistic fuzzy score functions. In 2022 17th Conference on Computer Science and Intelligence Systems (FedCSIS) (pp. 789-798). IEEE.
        Reference value: Kizielewicz, B., Paradowski, B., Więckowski, J., & Sałabun, W. (2022, September). Towards the identification of MARCOS models based on intuitionistic fuzzy score functions. In 2022 17th Conference on Computer Science and Intelligence Systems (FedCSIS) (pp. 789-798). IEEE.
    """

    x = np.array([0.17125, 0.21033])
    calculated_value = chen_score_1(x)
    reference_value = -0.039

    assert np.round(calculated_value, 3) == reference_value

def test_supriya_score():
    """
        Test veryfing correctness of the Supriya score formula.
        Formula: Kizielewicz, B., Paradowski, B., Więckowski, J., & Sałabun, W. (2022, September). Towards the identification of MARCOS models based on intuitionistic fuzzy score functions. In 2022 17th Conference on Computer Science and Intelligence Systems (FedCSIS) (pp. 789-798). IEEE.
        Reference value: Kizielewicz, B., Paradowski, B., Więckowski, J., & Sałabun, W. (2022, September). Towards the identification of MARCOS models based on intuitionistic fuzzy score functions. In 2022 17th Conference on Computer Science and Intelligence Systems (FedCSIS) (pp. 789-798). IEEE.
    """

    x = np.array([0.17125, 0.21033])
    calculated_value = supriya_score(x)
    reference_value = 0.041

    assert np.round(calculated_value, 3) == reference_value

def test_kharal_score_1():
    """
        Test veryfing correctness of the Kharal score 1 formula.
        Formula: Kizielewicz, B., Paradowski, B., Więckowski, J., & Sałabun, W. (2022, September). Towards the identification of MARCOS models based on intuitionistic fuzzy score functions. In 2022 17th Conference on Computer Science and Intelligence Systems (FedCSIS) (pp. 789-798). IEEE.
        Reference value: Kizielewicz, B., Paradowski, B., Więckowski, J., & Sałabun, W. (2022, September). Towards the identification of MARCOS models based on intuitionistic fuzzy score functions. In 2022 17th Conference on Computer Science and Intelligence Systems (FedCSIS) (pp. 789-798). IEEE.
    """

    x = np.array([0.17125, 0.21033])
    calculated_value = kharal_score_1(x)
    reference_value = -0.243

    assert np.round(calculated_value, 3) == reference_value

def test_kharal_score_2():
    """
        Test veryfing correctness of the Kharal score 2 formula.
        Formula: Kizielewicz, B., Paradowski, B., Więckowski, J., & Sałabun, W. (2022, September). Towards the identification of MARCOS models based on intuitionistic fuzzy score functions. In 2022 17th Conference on Computer Science and Intelligence Systems (FedCSIS) (pp. 789-798). IEEE.
        Reference value: Kizielewicz, B., Paradowski, B., Więckowski, J., & Sałabun, W. (2022, September). Towards the identification of MARCOS models based on intuitionistic fuzzy score functions. In 2022 17th Conference on Computer Science and Intelligence Systems (FedCSIS) (pp. 789-798). IEEE.
    """

    x = np.array([0.17125, 0.21033])
    calculated_value = kharal_score_2(x)
    reference_value = -0.428

    assert np.round(calculated_value, 3) == reference_value

def test_chen_score_2():
    """
        Test veryfing correctness of the Chen score 2 formula.
        Formula: Kizielewicz, B., Paradowski, B., Więckowski, J., & Sałabun, W. (2022, September). Towards the identification of MARCOS models based on intuitionistic fuzzy score functions. In 2022 17th Conference on Computer Science and Intelligence Systems (FedCSIS) (pp. 789-798). IEEE.
        Reference value: Kizielewicz, B., Paradowski, B., Więckowski, J., & Sałabun, W. (2022, September). Towards the identification of MARCOS models based on intuitionistic fuzzy score functions. In 2022 17th Conference on Computer Science and Intelligence Systems (FedCSIS) (pp. 789-798). IEEE.
    """

    x = np.array([0.17125, 0.21033])
    calculated_value = chen_score_2(x)
    reference_value = 0.480

    assert np.round(calculated_value, 3) == reference_value

def test_thakur_score():
    """
        Test veryfing correctness of the Thakur score formula.
        Formula: Thakur, P., Kizielewicz, B., Gandotra, N., Shekhovtsov, A., Saini, N., Saeid, A. B., & Sałabun, W. (2021). A New Entropy Measurement for the Analysis of Uncertain Data in MCDA Problems Using Intuitionistic Fuzzy Sets and COPRAS Method. Axioms, 10(4), 335.
        Reference value: Thakur, P., Kizielewicz, B., Gandotra, N., Shekhovtsov, A., Saini, N., Saeid, A. B., & Sałabun, W. (2021). A New Entropy Measurement for the Analysis of Uncertain Data in MCDA Problems Using Intuitionistic Fuzzy Sets and COPRAS Method. Axioms, 10(4), 335.
    """

    x = np.array([0.19028, 0.82374])
    calculated_value = thakur_score(x)
    reference_value = -0.6423

    assert np.round(calculated_value, 4) == reference_value

def test_zhang_xu_score_2():
    """
        Test veryfing correctness of the Zhang Xu score 2 formula.
        Formula: Pérez-Domínguez, L., Alvarado-Iniesta, A., Rodríguez-Borbón, I., & Vergara-Villegas, O. (2015). Intuitionistic fuzzy MOORA for supplier selection. Dyna, 82(191), 34-41.
        Reference value: Self-calculated empirical verification
    """

    x = np.array([0.609, 0.301, 0.09])
    calculated_value = zhang_xu_score_2(x)
    reference_value = 0.570

    assert np.round(calculated_value, 3) == reference_value
