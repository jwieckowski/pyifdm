# Copyright (c) 2023 Bartłomiej Kizielewicz

import unittest
import numpy as np
from pyifdm.similarity import *


class TestSimilarityFunctions(unittest.TestCase):

    def test_chen_similarity(self):
        """
            Test veryfing correctness of the similarity functions.
            Formula: Chen, S. M. (1995). Measures of similarity between vague sets. Fuzzy sets and Systems, 74(2), 217-223.
            Reference value: Ye, J. (2011). Cosine similarity measures for intuitionistic fuzzy sets and their applications. Mathematical and computer modelling, 53(1-2), 91-97.
        """
        a = np.array([0.3, 0.3])
        b = np.array([0.4, 0.4])
        self.assertAlmostEqual(chen_similarity(a, b), 0.85, places=2)

    def test_hong_kim_similarity(self):
        """
            Test veryfing correctness of the similarity functions.
            Formula: Hong, D. H., & Kim, C. (1999). A note on similarity measures between vague sets and between elements. Information sciences, 115(1-4), 83-96.
            Reference value: Ye, J. (2011). Cosine similarity measures for intuitionistic fuzzy sets and their applications. Mathematical and computer modelling, 53(1-2), 91-97.
        """
        a = np.array([0.3, 0.3])
        b = np.array([0.4, 0.4])
        self.assertAlmostEqual(hong_kim_similarity(a, b), 0.85, places=2)

    def test_li_xu_similarity(self):
        """
            Test veryfing correctness of the similarity functions.
            Formula: Li F, Xu Z (2001) Similarity measures between vague sets. J Softw 12(6):922–927
            Reference value: Ye, J. (2011). Cosine similarity measures for intuitionistic fuzzy sets and their applications. Mathematical and computer modelling, 53(1-2), 91-97.
        """
        a = np.array([0.3, 0.3])
        b = np.array([0.4, 0.4])
        self.assertAlmostEqual(li_xu_similarity(a, b), 0.9, places=2)

    def test_fan_zhang_similarity(self):
        """
            Test veryfing correctness of the similarity functions.
            Formula: Fan L, Zhang YX (2001) Similarity measures between vague sets. Chin J Softw 12(6):922–927
            Reference value: Ye, J. (2011). Cosine similarity measures for intuitionistic fuzzy sets and their applications. Mathematical and computer modelling, 53(1-2), 91-97.
        """
        a = np.array([1, 0])
        b = np.array([0, 0])
        self.assertAlmostEqual(fan_zhang_similarity(a, b), 1.0, places=2)

    def test_li_similarity(self):
        """
            Test veryfing correctness of the similarity functions.
            Formula: Li, Y., Zhongxian, C., & Degin, Y. (2002). Similarity measures between vague sets and vague entropy. J Comput Sci, 29(12), 129-132.
            Reference value: Ye, J. (2011). Cosine similarity measures for intuitionistic fuzzy sets and their applications. Mathematical and computer modelling, 53(1-2), 91-97.
        """
        a = np.array([0.3, 0.3])
        b = np.array([0.4, 0.4])
        self.assertAlmostEqual(li_similarity(a, b), 0.94, places=2)

    def test_ye_similarity(self):
        """
            Test veryfing correctness of the similarity functions.
            Formula: Ye, J. (2011). Cosine similarity measures for intuitionistic fuzzy sets and their applications. Mathematical and computer modelling, 53(1-2), 91-97.
            Reference value: Ye, J. (2011). Cosine similarity measures for intuitionistic fuzzy sets and their applications. Mathematical and computer modelling, 53(1-2), 91-97.
        """
        a = np.array([0.4, 0.2])
        b = np.array([0.5, 0.3])
        self.assertAlmostEqual(ye_similarity(a, b), 0.99, places=2)