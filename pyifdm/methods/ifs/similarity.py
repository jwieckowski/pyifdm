# Copyright (c) 2023 Bartłomiej Kizielewicz

import numpy as np

__all__ = [
    'chen_similarity',
    'fan_zhang_similarity',
    'hong_kim_similarity',
    'li_similarity',
    'li_xu_similarity',
    'ye_similarity'
]


def chen_similarity(a, b):
    """
        Calculates similarity of the Intuitionistic Fuzzy Sets (u1, v1)|(u2, v2) and returns a crisp value.

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Set (u1, v1)
            b : ndarray
                Intuitionistic Fuzzy Set (u2, v2)

        Returns
        -------
            float
                Crisp value
    """
    # cast types
    a = a.astype(float)
    b = b.astype(float)

    if a.ndim == 1 and b.ndim == 1:
        return 1 - (np.sum(np.abs(a[0] - a[1]) - np.abs(b[0] - b[1])) / 2 * a.ndim)
    elif a.ndim == 2 and b.ndim == 2:
        return 1 - (np.sum(np.abs(a[:, 0] - a[:, 1]) - np.abs(b[:, 0] - b[:, 1])) / 2 * a.ndim)
    else:
        return 1 - (np.sum(np.abs(a[:, :, 0] - a[:, :, 1]) - np.abs(b[:, :, 0] - b[:, :, 1])) / 2 * a.ndim)


def hong_kim_similarity(a, b):
    """
        Calculates similarity of the Intuitionistic Fuzzy Sets (u1, v1)|(u2, v2) and returns a crisp value.

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Set (u1, v1)
            b : ndarray
                Intuitionistic Fuzzy Set (u2, v2)

        Returns
        -------
            float
                Crisp value
    """
    # cast types
    a = a.astype(float)
    b = b.astype(float)

    if a.ndim == 1 and b.ndim == 1:
        return 1 - (np.sum(np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])) / 2 * a.ndim)
    elif a.ndim == 2 and b.ndim == 2:
        return 1 - (np.sum(np.abs(a[:, 0] - b[:, 0]) + np.abs(a[:, 1] - b[:, 1])) / 2 * a.ndim)
    else:
        return 1 - (np.sum(np.abs(a[:, :, 0] - b[:, :, 0]) + np.abs(a[:, :, 1] - b[:, :, 1])) / 2 * a.ndim)


def li_xu_similarity(a, b):
    """
        Calculates similarity of the Intuitionistic Fuzzy Sets (u1, v1)|(u2, v2) and returns a crisp value.

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Set (u1, v1)
            b : ndarray
                Intuitionistic Fuzzy Set (u2, v2)

        Returns
        -------
            float
                Crisp value
    """
    # cast types
    a = a.astype(float)
    b = b.astype(float)

    if a.ndim == 1 and b.ndim == 1:
        return 1 - np.sum(np.abs((a[0] - a[1]) - (b[0] - b[1]))) / 4 * a.ndim - np.sum(
            np.abs(a[0] - a[1]) + np.abs(b[0] - b[1])) / 4 * a.ndim
    elif a.ndim == 2 and b.ndim == 2:
        return 1 - np.sum(np.abs((a[:, 0] - a[:, 1]) - (b[:, 0] - b[:, 1]))) / 4 * a.ndim - np.sum(
            np.abs(a[:, 0] - a[:, 1]) + np.abs(b[:, 0] - b[:, 1])) / 4 * a.ndim
    else:
        return 1 - np.sum(np.abs((a[:, :, 0] - a[:, 1]) - (b[:, :, 0] - b[:, :, 1]))) / 4 * a.ndim - np.sum(
            np.abs(a[:, :, 0] - a[:, :, 1]) + np.abs(b[:, :, 0] - b[:, :, 1])) / 4 * a.ndim


def fan_zhang_similarity(a, b):
    """
        Calculates similarity of the Intuitionistic Fuzzy Sets (u1, v1)|(u2, v2) and returns a crisp value.

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Set (u1, v1)
            b : ndarray
                Intuitionistic Fuzzy Set (u2, v2)

        Returns
        -------
            float
                Crisp value
    """
    # cast types
    a = a.astype(float)
    b = b.astype(float)

    if a.ndim == 1 and b.ndim == 1:
        return 1 - np.sum(np.abs((a[0] - a[1]) - (b[0] - b[1])) + np.abs((a[0] - b[0]) - (a[1] - b[1]))) / 4 * a.ndim
    elif a.ndim == 2 and b.ndim == 2:
        return 1 - np.sum(np.abs((a[:, 0] - a[:, 1]) - (b[:, 0] - b[:, 1])) + np.abs(
            (a[:, 0] - b[:, 0]) - (a[:, 1] - b[:, 1]))) / 4 * a.ndim
    else:
        return 1 - np.sum(np.abs((a[:, :, 0] - a[:, :, 1]) - (b[:, :, 0] - b[:, :, 1])) + np.abs(
            (a[:, :, 0] - b[:, :, 0]) - (a[:, :, 1] - b[:, :, 1]))) / 4 * a.ndim


def li_similarity(a, b):
    """
        Calculates similarity of the Intuitionistic Fuzzy Sets (u1, v1)|(u2, v2) and returns a crisp value.

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Set (u1, v1)
            b : ndarray
                Intuitionistic Fuzzy Set (u2, v2)

        Returns
        -------
            float
                Crisp value
    """
    # cast types
    a = a.astype(float)
    b = b.astype(float)

    if a.ndim == 1 and b.ndim == 1:
        return 1 - np.sqrt(np.sum((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / 2 * a.ndim)
    elif a.ndim == 2 and b.ndim == 2:
        return 1 - np.sqrt(np.sum((a[:, 0] - b[:, 0]) ** 2 + (a[:, 1] - b[:, 1]) ** 2) / 2 * a.ndim)
    else:
        return 1 - np.sqrt(np.sum((a[:, :, 0] - b[:, :, 0]) ** 2 + (a[:, :, 1] - b[:, :, 1]) ** 2) / 2 * a.ndim)


def ye_similarity(a, b):
    """
        Calculates similarity of the Intuitionistic Fuzzy Sets (u1, v1)|(u2, v2) and returns a crisp value.

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Set (u1, v1)
            b : ndarray
                Intuitionistic Fuzzy Set (u2, v2)

        Returns
        -------
            float
                Crisp value
    """
    # cast types
    a = a.astype(float)
    b = b.astype(float)

    if a.ndim == 1 and b.ndim == 1:
        return np.sum(
            (a[0] * b[0] + a[1] * b[1]) / (np.sqrt(a[0] ** 2 + a[1] ** 2) * np.sqrt(b[0] ** 2 + b[1] ** 2))) / a.ndim
    elif a.ndim == 2 and b.ndim == 2:
        return np.sum(
            (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]) / (np.sqrt(a[:, 0] ** 2 + a[:, 1] ** 2) * np.sqrt(b[:, 0] ** 2 + b[:, 1] ** 2))) / a.ndim
    else:
        return np.sum(
            (a[:, :, 0] * b[:, :, 0] + a[:, :, 1] * b[:, :, 1]) / (np.sqrt(a[:, :, 0] ** 2 + a[:, :, 1] ** 2) * np.sqrt(b[:, :, 0] ** 2 + b[:, :, 1] ** 2))) / a.ndim
