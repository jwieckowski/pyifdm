# Copyright (c) 2022 Jakub Więckowski

from .mabac.ifs import ifs
from .ifs.normalization import swap_normalization
from .ifs.score import liu_wang_score
from .ifs.distance import luo_distance

from .validator import Validator


class ifMABAC():
    def __init__(self, normalization=swap_normalization, distance=luo_distance, score=liu_wang_score, p=2.25, g=0.88):
        """
        Create Intuitionistic Fuzzy MAIRCA method object with normalization and score functions

        Parameters
        ----------
            normalization: callable, default=swap_normalization
                Function used to normalize the decision matrix

            distance: callable, default=lou_distance
                Function used to calculate distance between two IFS

            score: callable, default=liu_wang_score
                Function used to calculate crisp score of IFS

            p: float, default=2.25
                Adjust parameter for distance calculation

            g: float, default=0.88
                Adjust parameter for distance calculation
        """

        self.normalization = normalization
        self.distance = distance
        self.score = score
        self.p = p
        self.g = g

    def __call__(self, matrix, weights, types):
        """
        Calculates the alternatives preferences

        Parameters
        ----------
            matrix : ndarray
                Decision matrix / alternatives data.
                Alternatives are in rows and Criteria are in columns.

            weights : ndarray
                Vector of criteria weights in a crisp or Intuitionistic Fuzzy form

            types : ndarray
                Types of criteria, 1 profit, -1 cost

        Returns
        ----------
            ndarray:
                Preference calculated for alternatives. Greater values are placed higher in ranking
        """
        # validate data
        Validator.ifs_validation(matrix, weights, types)

        return ifs(matrix, weights, types, self.normalization, self.distance, self.score, self.p, self.g).astype(float)
