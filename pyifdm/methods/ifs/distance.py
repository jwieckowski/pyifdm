# Copyright (c) 2022-2023 Jakub Więckowski

import numpy as np

__all__ = [
    'euclidean_distance',
    'grzegorzewski_distance',
    'hamming_distance',
    'hausdorf_euclidean_distance',
    'luo_distance',
    'normalized_euclidean_distance',
    'normalized_hamming_distance',
    'wang_xin_distance_1',
    'wang_xin_distance_2',
    'yang_chiclana_distance'
]


def euclidean_distance(a, b):
    """
        Calculates the distance between two Intuitionistic Fuzzy Sets (u, v) using Euclidean distance

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Sets (u, v)

            b : ndarray
                Intuitionistic Fuzzy Sets (u, v)

        Returns
        -------
            float
                Crisp value representing distance
    """
    
    if len(a) == 2 or len(b) == 2:
        ap = 1 - a[0] - a[1]
        bp = 1 - b[0] - b[1]
    else:
        ap, bp = a[2], b[2]

    return np.sqrt(((a[0] - b[0])**2 + (a[1] - b[1])**2 + (ap - bp)**2) / 2)

def grzegorzewski_distance(a, b):
    """
        Calculates the distance between two Intuitionistic Fuzzy Sets (u, v) using Grzegorzewski distance

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Sets (u, v)

            b : ndarray
                Intuitionistic Fuzzy Sets (u, v)

        Returns
        -------
            float
                Crisp value representing distance
    """

    return np.max([np.abs(a[0] - b[0]), np.abs(a[1] - b[1])])

def hamming_distance(a, b):
    """
        Calculates the distance between two Intuitionistic Fuzzy Sets (u, v) using Hamming distance

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Sets (u, v)

            b : ndarray
                Intuitionistic Fuzzy Sets (u, v)

        Returns
        -------
            float
                Crisp value representing distance
    """

    if len(a) == 2 or len(b) == 2:
        ap = 1 - a[0] - a[1]
        bp = 1 - b[0] - b[1]
    else:
        ap, bp = a[2], b[2]

    return (np.abs(a[0] - b[0]) + np.abs(a[1] - b[1]) + np.abs(ap - bp)) / 2

def hausdorf_euclidean_distance(a, b):
    """
        Calculates the distance between two Intuitionistic Fuzzy Sets (u, v) using Hausdorf measure-based Euclidean distance

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Sets (u, v)

            b : ndarray
                Intuitionistic Fuzzy Sets (u, v)

        Returns
        -------
            float
                Crisp value representing distance
    """

    return np.max([(a[0] - b[0])**2, (a[1] - b[1])**2])

def luo_distance(a, b):
    """
        Calculates the distance between two Intuitionistic Fuzzy Sets (u, v) using Luo distance

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Sets (u, v)

            b : ndarray
                Intuitionistic Fuzzy Sets (u, v)

        Returns
        -------
            float
                Crisp value representing distance
    """

    if len(a) == 2 or len(b) == 2:
        ap = 1 - a[0] - a[1]
        bp = 1 - b[0] - b[1]
    else:
        ap, bp = a[2], b[2]

    l1 = (np.abs(a[0] - b[0]) + np.abs(a[1] - b[1]) + np.abs((a[0] + 1 - a[1]) - (b[0] + 1 - b[1]))) / 2
    l2 = (ap - bp) / 2
    l3 = np.max([np.abs(a[0] - b[0]), np.abs(a[1] - b[1]), np.abs(ap - bp)/2])
    return 1/6 * (l1 + l2 + l3)

def normalized_euclidean_distance(a, b):
    """
        Calculates the distance between two Intuitionistic Fuzzy Sets (u, v) using normalized Euclidean distance

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Sets (u, v)

            b : ndarray
                Intuitionistic Fuzzy Sets (u, v)

        Returns
        -------
            float
                Crisp value representing distance
    """

    if len(a) == 2 or len(b) == 2:
        ap = 1 - a[0] - a[1]
        bp = 1 - b[0] - b[1]
    else:
        ap, bp = a[2], b[2]

    return ((a[0] - b[0])**2 + (a[1] - b[1])**2 + (ap - bp)**2)

def normalized_hamming_distance(a, b):
    """
        Calculates the distance between two Intuitionistic Fuzzy Sets (u, v) using normalized Hamming distance

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Sets (u, v)

            b : ndarray
                Intuitionistic Fuzzy Sets (u, v)

        Returns
        -------
            float
                Crisp value representing distance
    """

    if len(a) == 2 or len(b) == 2:
        ap = 1 - a[0] - a[1]
        bp = 1 - b[0] - b[1]
    else:
        ap, bp = a[2], b[2]

    return (np.abs(a[0] - b[0]) + np.abs(a[1] - b[1]) + np.abs(ap - bp))

def wang_xin_distance_1(a, b):
    """
        Calculates the distance between two Intuitionistic Fuzzy Sets (u, v) using Wang Xin distance 1

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Sets (u, v)

            b : ndarray
                Intuitionistic Fuzzy Sets (u, v)

        Returns
        -------
            float
                Crisp value representing distance
    """
    
    return (np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])) / 4 + np.max([np.abs(a[0] - b[0]), np.abs(a[1] - b[1])]) / 2

def wang_xin_distance_2(a, b):
    """
        Calculates the distance between two Intuitionistic Fuzzy Sets (u, v) using Wang Xin distance 2

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Sets (u, v)

            b : ndarray
                Intuitionistic Fuzzy Sets (u, v)

        Returns
        -------
            float
                Crisp value representing distance
    """

    return (np.abs(a[0] - b[0])/2 + np.abs(a[1] - b[1])/2)

def yang_chiclana_distance(a, b):
    """
        Calculates the distance between two Intuitionistic Fuzzy Sets (u, v) using Yang & Chiclana distance

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Sets (u, v)

            b : ndarray
                Intuitionistic Fuzzy Sets (u, v)

        Returns
        -------
            float
                Crisp value representing distance
    """
    
    if len(a) == 2 or len(b) == 2:
        ap = 1 - a[0] - a[1]
        bp = 1 - b[0] - b[1]
    else:
        ap, bp = a[2], b[2]

    return np.max([np.abs(a[0] - b[0]), np.abs(a[1] - b[1]) * np.abs(ap - bp)])

