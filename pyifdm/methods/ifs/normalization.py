# Copyright (c) 2022-2023 Jakub Więckowski

import numpy as np

__all__ = [
    'ecer_normalization',
    'max_normalization',
    'minmax_normalization',
    'supriya_normalization',
    'swap_normalization',
]

def ecer_normalization(matrix, types):
    """
        Calculates the normalized value of Intuitionistic Fuzzy matrix using Ecer normalization

        Parameters
        ----------
            matrix : ndarray
                Matrix with Intuitionistic Fuzzy Sets

            types : ndarray
                Types of criteria, 1 profit, -1 cost

        Returns
        -------
            ndarray
                Normalized Intuitionistic Fuzzy matrix
    """
    nmatrix = matrix.copy()

    # validate data
    if isinstance(np.max(matrix[:, types==1], axis=0)[0], float):
        if any([m == 0 for m in np.max(matrix[:, types==1], axis=0)]):
            raise ValueError('Maximum value in matrix cannot equal 0')
    else:
        if any([any(m == 0) for m in np.max(matrix[:, types==1], axis=0)]):
            raise ValueError('Maximum value in matrix cannot equal 0')

    nmatrix[:, types==1] = matrix[:, types==1] / np.max(matrix[:, types==1], axis=0)
    nmatrix[:, types==-1] = np.min(matrix[:, types==-1], axis=0) / matrix[:, types==-1] 
    
    return nmatrix.astype(float)

def max_normalization(matrix, types):
    """
        Calculates the normalized value of Intuitionistic Fuzzy matrix using Max normalization

        Parameters
        ----------
            matrix : ndarray
                Matrix with Intuitionistic Fuzzy Sets

            types : ndarray
                Types of criteria, 1 profit, -1 cost

        Returns
        -------
            ndarray
                Normalized Intuitionistic Fuzzy matrix
    """

    nmatrix = matrix.copy()
    if 1 in types:
        nmatrix[:, types == 1] = matrix[:, types == 1] / np.max(([np.max(matrix[:, types == 1, 0]), np.min(matrix[:, types == 1, 1])]))
    if -1 in types:
        nmatrix[:, types == -1] = np.min(([np.min(matrix[:, types == -1, 0]), np.max(matrix[:, types == 1, 1])])) / matrix[:, types == -1]

    return nmatrix.astype(float)

def minmax_normalization(matrix, types):
    """
        Calculates the normalized value of Intuitionistic Fuzzy matrix using Min-Max normalization

        Parameters
        ----------
            matrix : ndarray
                Matrix with Intuitionistic Fuzzy Sets

            types : ndarray
                Types of criteria, 1 profit, -1 cost

        Returns
        -------
            ndarray
                Normalized Intuitionistic Fuzzy matrix
    """

    cmax = np.max(matrix, axis=0)
    cmin = np.min(matrix, axis=0)

    # validate data
    if np.min(cmax[types == 1] - cmin[types == 1]) == 0 or np.min(cmax[types == -1] - cmin[types == -1]) == 0:
        raise ValueError('Subtraction result of matrix elements cannot equal 0')

    nmatrix = np.zeros((matrix.shape))
    nmatrix[:, types == 1] = (matrix[:, types == 1] - cmin[types == 1]) / (cmax[types == 1] - cmin[types == 1])
    nmatrix[:, types == -1] = (cmax[types == -1] - matrix[:, types == -1]) / (cmax[types == -1] - cmin[types == -1])

    return nmatrix.astype(float)

def supriya_normalization(matrix, *args):
    """
        Calculates the normalized value of Intuitionistic Fuzzy matrix using Supriya normalization

        Parameters
        ----------
            matrix : ndarray
                Matrix with Intuitionistic Fuzzy Sets

            *args : 
                Additional parameters

        Returns
        -------
            ndarray
                Normalized Intuitionistic Fuzzy matrix
    """
    nmatrix = np.zeros(matrix.shape, dtype=object)

    nmatrix[:, :, 0] = matrix[:, :, 0] / np.max(matrix[:, :, 0], axis=0)
    nmatrix[:, :, 1] = (matrix[:, :, 1] - np.min(matrix[:, :, 1], axis=0)) / (1 - np.min(matrix[:, :, 1], axis=0))

    return nmatrix.astype(float)

def swap_normalization(matrix, types):
    """
        Calculates the normalized value of Intuitionistic Fuzzy matrix using Swap normalization

        Parameters
        ----------
            matrix : ndarray
                Matrix with Intuitionistic Fuzzy Sets

            types : ndarray
                Types of criteria, 1 profit, -1 cost

        Returns
        -------
            ndarray
                Normalized Intuitionistic Fuzzy matrix
    """
    nmatrix = matrix.copy()

    nmatrix[:, types==-1, 0], nmatrix[:, types==-1, 1] = nmatrix[:, types==-1, 1], nmatrix[:, types==-1, 0]
    
    return nmatrix.astype(float)
