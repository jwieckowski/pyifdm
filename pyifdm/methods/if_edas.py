# Copyright (c) 2022 Jakub Więckowski

from .edas.ifs import ifs
from .ifs.normalization import swap_normalization
from .ifs.score import liu_wang_score
from ..helpers import rank

from .validator import Validator


class ifEDAS():
    def __init__(self, normalization=swap_normalization, score=liu_wang_score):
        """
        Create Intuitionistic Fuzzy EDAS method object with normalization function

        Parameters
        ----------
            normalization: callable, default=swap_normalization
                Function used to calculate normalized decision matrix

            score: callable, default=liu_wang_score
                Function used to calculate crisp score of IFS

        """

        self.normalization = normalization
        self.score = score
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
                Types of criteria, 1 profit, -1 cost

        Returns
        ----------
            ndarray:
                Preference calculated for alternatives. Greater values are placed higher in ranking
        """
        # validate data
        Validator.ifs_validation(matrix, weights, types)

        self.preferences = ifs(matrix, weights, types, self.normalization, self.score).astype(float)
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