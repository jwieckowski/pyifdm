# Copyright (c) 2023 Bartłomiej Kizielewicz, Jakub Więckowski

import unittest
import numpy as np
from pyifdm.methods.ifs.similarity import *


def test_chen_similarity():
    """
        Test veryfing correctness of the similarity functions.
        Formula: Chen, S. M. (1995). Measures of similarity between vague sets. Fuzzy sets and Systems, 74(2), 217-223.
        Reference value: Ye, J. (2011). Cosine similarity measures for intuitionistic fuzzy sets and their applications. Mathematical and computer modelling, 53(1-2), 91-97.
    """
    a = np.array([0.8, 0.15])
    b = np.array([0.7, 0.28])
    assert np.round(chen_similarity(a, b), 3) == 0.885

def test_hong_kim_similarity():
    """
        Test veryfing correctness of the similarity functions.
        Formula: Hong, D. H., & Kim, C. (1999). A note on similarity measures between vague sets and between elements. Information sciences, 115(1-4), 83-96.
        Reference value: Ye, J. (2011). Cosine similarity measures for intuitionistic fuzzy sets and their applications. Mathematical and computer modelling, 53(1-2), 91-97.
    """
    a = np.array([0.8, 0.15])
    b = np.array([0.7, 0.28])
    assert np.round(hong_kim_similarity(a, b), 3) == 0.885

def test_li_xu_similarity():
    """
        Test veryfing correctness of the similarity functions.
        Formula: Li F, Xu Z (2001) Similarity measures between vague sets. J Softw 12(6):922–927
        Reference value: Ye, J. (2011). Cosine similarity measures for intuitionistic fuzzy sets and their applications. Mathematical and computer modelling, 53(1-2), 91-97.
    """
    a = np.array([0.8, 0.15])
    b = np.array([0.7, 0.28])
    assert np.round(li_xu_similarity(a, b), 3) == 0.675

def test_fan_zhang_similarity():
    """
        Test veryfing correctness of the similarity functions.
        Formula: Fan L, Zhang YX (2001) Similarity measures between vague sets. Chin J Softw 12(6):922–927
        Reference value: Ye, J. (2011). Cosine similarity measures for intuitionistic fuzzy sets and their applications. Mathematical and computer modelling, 53(1-2), 91-97.
    """
    a = np.array([0.8, 0.15])
    b = np.array([0.7, 0.28])
    assert np.round(fan_zhang_similarity(a, b), 3)  == 0.885

def test_li_similarity():
    """
        Test veryfing correctness of the similarity functions.
        Formula: Li, Y., Zhongxian, C., & Degin, Y. (2002). Similarity measures between vague sets and vague entropy. J Comput Sci, 29(12), 129-132.
        Reference value: Ye, J. (2011). Cosine similarity measures for intuitionistic fuzzy sets and their applications. Mathematical and computer modelling, 53(1-2), 91-97.
    """
    a = np.array([0.8, 0.15])
    b = np.array([0.7, 0.28])
    assert np.round(li_similarity(a, b), 3) == 0.884

def test_ye_similarity():
    """
        Test veryfing correctness of the similarity functions.
        Formula: Ye, J. (2011). Cosine similarity measures for intuitionistic fuzzy sets and their applications. Mathematical and computer modelling, 53(1-2), 91-97.
        Reference value: Ye, J. (2011). Cosine similarity measures for intuitionistic fuzzy sets and their applications. Mathematical and computer modelling, 53(1-2), 91-97.
    """
    a = np.array([0.8, 0.15])
    b = np.array([0.7, 0.28])
    assert np.round(ye_similarity(a, b), 3) == 0.981