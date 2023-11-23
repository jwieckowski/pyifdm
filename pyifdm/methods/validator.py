# Copyright (c) 2022-2023 Jakub Więckowski

import numpy as np

class Validator():

    @staticmethod
    def validate_input(matrix, weights, types):
        """
        Checks if number of criteria, number of weights, and number of types are the same

        Parameters
        ----------
            matrix : ndarray
                Decision matrix / alternatives data.
                Alternatives are in rows and Criteria are in columns.

            weights : ndarray
                Vector of weights in a crisp form

            types : ndarray
                Types of criteria, 1 profit, -1 cost

        Returns
        -------
            raises:
                ValueError if shapes of matrix, weights and types are not the same

        """

        if len(np.unique([matrix.shape[1], weights.shape[0], types.shape[0]])) != 1:
            raise ValueError(f'Number of criteria should equals number of weights and types, not {matrix.shape[1]}, {weights.shape[0]}, {types.shape[0]}')
    
    @staticmethod
    def validate_ifs_matrix(matrix):
        """
        Checks if IFS matrix is defined properly, all elements should have length of 2 or 3

        Parameters
        ----------
            matrix : ndarray
                Decision matrix / alternatives data.
                Alternatives are in rows and Criteria are in columns.

        Returns
        -------
            raises:
                ValueError if matrix elements has different length than 2 or 3

        """

        if matrix.ndim != 3 or (matrix.shape[2] != 2 and matrix.shape[2] != 3):
            raise ValueError(
                'IFS matrix elements should all have length of 2 or 3')

    @staticmethod
    def validate_weights(weights, crisp_weights=False):
        """
        For crisp weights checks if sum of weights equals 1
        For fuzzy weights checks if given as IFS

        Parameters
        ----------
            weights : ndarray
                Vector of weights in a crisp form or as a IFS array

            crisp_weights : bool
                Flag to check if only crisp weights are acceptable

        Returns
        -------
            raises:
                ValueError if sum of weights is different than 1 or not in IFS form

        """

        if crisp_weights:
            if np.array(weights).ndim != 1:
                raise ValueError('Weights should be given as crisp values')

        if isinstance(weights[0], np.float_):
            if np.round(np.sum(weights), 3) != 1:
                raise ValueError(
                    f'Sum of crisp weights should equal 1, not {np.sum(weights)}')
        else:
            if len(list(set([len(w) for w in weights]))) != 1 and len(list(set([len(w) != 3 for w in weights]))) != 1:
                raise ValueError(
                    'Intuitionistic Fuzzy weights should all have the same length')

    @staticmethod
    def validate_types(types):
        """
        Checks if all criteria types are same type

        Parameters
        ----------
            types : ndarray
                Types of criteria, 1 profit, -1 cost

        Returns
        -------
            raises:
                ValueError if criteria types are the same

        """

        if len(np.unique(types)) == 1:
            raise ValueError('Criteria types should not be the same')

    @staticmethod
    def ifs_validation(matrix, weights, types, mixed_types=False, crisp_weights=False):
        """
        Runs all validations for the fuzzy IFS extension

        Parameters
        ----------
            matrix : ndarray
                Decision matrix / alternatives data.
                Alternatives are in rows and Criteria are in columns.

            weights : ndarray
                Vector of weights in a crisp form

            types : ndarray
                Types of criteria, 1 profit, -1 cost

            mixed_types : boolean, default=False
                Flag to determine if types array must be mixed

            crisp_weights : bool
                Flag to check if only crisp weights are acceptable

        Returns
        -------
            raises:
                ValueError if one of validations do not pass

        """
        Validator.validate_input(matrix, weights, types)
        Validator.validate_ifs_matrix(matrix)
        Validator.validate_weights(weights, crisp_weights)
        if mixed_types:
            Validator.validate_types(types)

