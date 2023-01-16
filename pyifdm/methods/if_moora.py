# Copyright (c) 2022 Jakub Więckowski

from .moora.ifs import ifs
from .ifs.score import zhang_xu_score_2

from .validator import Validator

class ifMOORA():
    def __init__(self, score=zhang_xu_score_2, normalization=None):
        """
        Create Intuitionistic Fuzzy MOORA method object with normalization and score functions

        Parameters
        ----------
            score: callable, default=zhang_xu_score_2
                Function used to calculate crisp score of IFS
            
            normalization: callable, default=None
                Function used to normalize the decision matrix

        """

        self.normalization = normalization
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
                Types of criteria, 1 profit, -1 cost.
                Criteria types cannot be all profit or all cost.

        Returns
        ----------
            ndarray:
                Preference calculated for alternatives. Greater values are placed higher in ranking
        """
        # validate data
        Validator.ifs_validation(matrix, weights, types, mixed_types=True)

        return ifs(matrix, weights, types, self.normalization, self.score).astype(float)
