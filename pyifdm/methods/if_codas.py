# Copyright (c) 2022 Jakub Więckowski

from .codas.ifs import ifs
from .ifs.normalization import swap_normalization
from .ifs.distance import euclidean_distance, hamming_distance

from .validator import Validator


class ifCODAS():
    def __init__(self, normalization=swap_normalization, distance_1=euclidean_distance, distance_2=hamming_distance, tau=0.05):
        """
        Create Intuitionistic Fuzzy CODAS method object with normalization function and distances metrics

        Parameters
        ----------
            normalization: callable, default=swap_normalization
                Function used to calculate normalized decision matrix

            distance_1: callable, default=euclidean_distance
                Function used to calculate distance between two IFS

            distance_2: callable, default=hamming_distance
                Function used to calculate distance between two IFS

            tau: float, default=0.05
                Threshold parameter

        """

        self.normalization = normalization
        self.distance_1 = distance_1
        self.distance_2 = distance_2
        self.tau = tau

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

        return ifs(matrix, weights, types, self.normalization, self.distance_1, self.distance_2, self.tau).astype(float)
