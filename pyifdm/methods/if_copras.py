# Copyright (c) 2022 Jakub Więckowski

from .copras.ifs import ifs
from .ifs.score import thakur_score
from ..helpers import rank

from .validator import Validator


class ifCOPRAS():
    def __init__(self, score=thakur_score, normalization=None):
        """
        Create Intuitionistic Fuzzy COPRAS method object with normalization and score functions

        Parameters
        ----------
            score: callable, default=thakur_score
                Function used to calculate crisp score of IFS
            
            normalization: callable, default=None
                Function used to calculate normalized decision matrix

        """

        self.score = score
        self.normalization = normalization
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