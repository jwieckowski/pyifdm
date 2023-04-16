# Copyright (c) 2022 Jakub Więckowski

import numpy as np
from .vikor.ifs import ifs
from .ifs.distance import hamming_distance
from ..helpers import rank

from .validator import Validator


class ifVIKOR():
    def __init__(self, distance=hamming_distance, normalization=None, v=0.5):
        """
        Creates Intuitionistic Fuzzy VIKOR method object with normalization and distance functions

        Parameters
        ----------
            distance: callable, default=hamming_distance
                Function used to calculate distance between two IFS
            
            normalization: callable, default=None
                Function used to normalize the decision matrix

            v : float, default=0.5
                Weight of the strategy (see VIKOR algorithm explanation).
        """

        self.normalization = normalization
        self.distance = distance
        self.v = v
        self.__descending = False

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
                Preference calculated for alternatives. Lower values are placed higher in ranking
        """
        # validate data
        Validator.ifs_validation(matrix, weights, types)

        self.preferences = ifs(matrix, weights, types, self.normalization, self.distance, self.v)
        return self.preferences

    def rank(self):
        """
            Calculates the alternatives ranking based on the obtained preferences

            Returns
            ----------
                ndarray:
                    Ranking of alternatives for the S, R, Q approaches
        """
        try:
            return np.array([rank(pref, self.__descending) for pref in self.preferences])
        except AttributeError:
            raise AttributeError('Cannot calculate ranking before assessment')
        except:
            raise ValueError('Error occurred in ranking calculation')