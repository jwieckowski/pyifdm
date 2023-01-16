# Copyright (c) 2022 Jakub Więckowski

from .mairca.ifs import ifs
from .ifs.normalization import minmax_normalization
from .ifs.distance import normalized_euclidean_distance
from .ifs.score import liu_wang_score

from .validator import Validator


class ifMAIRCA():
    def __init__(self, normalization=minmax_normalization, distance=normalized_euclidean_distance, score=liu_wang_score):
        """
        Create Intuitionistic Fuzzy MAIRCA method object with normalization and distance functions

        Parameters
        ----------
            normalization: callable, default=minmax_normalization
                Function used to normalize the decision matrix

            distance: callable, default=normalized_euclidean_distance
                Function used to calculate distance between two IFS

            score: callable, default=liu_wang_score
                Function used to calculate crisp score of IFS

        """

        self.normalization = normalization
        self.distance = distance
        self.score = score

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

        return ifs(matrix, weights, types, self.normalization, self.distance, self.score).astype(float)
