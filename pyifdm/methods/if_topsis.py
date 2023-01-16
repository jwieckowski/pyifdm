# Copyright (c) 2022 Jakub Więckowski

from .topsis.ifs import ifs
from .ifs.distance import normalized_euclidean_distance

from .validator import Validator


class ifTOPSIS():
    def __init__(self, distance=normalized_euclidean_distance, normalization=None):
        """
        Creates Intuitionistic Fuzzy TOPSIS method object with normalization and distance functions

        Parameters
        ----------
            distance: callable, default=normalized_euclidean_distance
                Function used to calculate distance between two IFS
            
            normalization: callable, default=None
                Function used to normalize the decision matrix

        """

        self.normalization = normalization
        self.distance = distance

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
                Types of criteria, 1 profit, -1 cost.
                Criteria types cannot be all profit or all cost.

        Returns
        ----------
            ndarray:
                Preference calculated for alternatives. Greater values are placed higher in ranking
        """
        # validate data
        Validator.ifs_validation(matrix, weights, types, mixed_types=True)

        return ifs(matrix, weights, types, self.normalization, self.distance).astype(float)
