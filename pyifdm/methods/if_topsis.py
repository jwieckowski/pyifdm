# Copyright (c) 2022 Jakub Więckowski

from .topsis.ifs import ifs
from .ifs.distance import normalized_euclidean_distance
from ..helpers import rank

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
        self.__descending = True

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

        self.preferences = ifs(matrix, weights, types, self.normalization, self.distance).astype(float)
        return self.preferences

    def rank(self):
        """
            Calculates the alternatives ranking based on the obtained preferences

            Returns
            ----------
                ndarray:
                    Ranking of alternatives
        """
        try:
            return rank(self.preferences, self.__descending)
        except AttributeError:
            raise AttributeError('Cannot calculate ranking before assessment')
        except:
            raise ValueError('Error occurred in ranking calculation')