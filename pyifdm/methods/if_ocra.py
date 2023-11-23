# Copyright (c) 2022 Jakub Więckowski

from .ocra.ifs import ifs
from .ifs.score import chen_score_1
from ..helpers import rank

from .validator import Validator

class ifOCRA():
    def __init__(self, score=chen_score_1):
        """
        Create Intuitionistic Fuzzy OCRA method object with normalization and score functions

        Parameters
        ----------
            score: callable, default=chen_score_1
                Function used to calculate crisp score of IFS

        """

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
                Types of criteria, 1 profit, -1 cost.
                Criteria types cannot be all profit or all cost.

        Returns
        ----------
            ndarray:
                Preference calculated for alternatives. Greater values are placed higher in ranking
        """
        # validate data
        Validator.ifs_validation(matrix, weights, types, crisp_weights=True)

        self.preferences = ifs(matrix, weights, types, self.score).astype(float)
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