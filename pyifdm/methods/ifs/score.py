# Copyright (c) 2022 Jakub Więckowski

import numpy as np

__all__ = [
    'chen_score_1',
    'chen_score_2',
    'kharal_score_1',
    'kharal_score_2',
    'liu_wang_score',
    'supriya_score',
    'thakur_score',
    'wan_dong_score_1',
    'wan_dong_score_2',
    'wei_score',
    'zhang_xu_score_1',
    'zhang_xu_score_2'
]

def chen_score_1(a):
    """
        Calculates score of the Intuitionistic Fuzzy Set (u, v) and returns a crisp value.
        Uses a formula: (u - v)

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Set (u, v)

        Returns
        -------
            float
                Crisp value
    """
    # cast types
    a = a.astype(float)

    if a.ndim == 1:
        return a[0] - a[1]
    elif a.ndim == 2:
        return a[:, 0] - a[:, 1]
    else:
        return a[:, :, 0] - a[:, :, 1]

def chen_score_2(a, y=0.5):
    """
        Calculates score of the Intuitionistic Fuzzy Set (u, v) and returns a crisp value.
        Uses a formula: (y * u + (1 - y) * (1 - v)) 

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Set (u, v)
            
            y : float, default=0.5
                Adjusting parameter

        Returns
        -------
            float
                Crisp value
    """
    # cast types
    a = a.astype(float)

    if a.ndim == 1:
        return y * a[0] + (1 - y) * (1 - a[1])
    elif a.ndim == 2:
        return y * a[:, 0] + (1 - y) * (1 - a[:, 1])
    else:
        return y * a[:, :, 0] + (1 - y) * (1 - a[:, :, 1])

def kharal_score_1(a):
    """
        Calculates score of the Intuitionistic Fuzzy Set (u, v) and returns a crisp value.
        Uses a formula: (u - (v + (1 - u - v)) / 2)

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Set (u, v)

        Returns
        -------
            float
                Crisp value
    """
    # cast types
    a = a.astype(float)

    if a.ndim == 1:
        return a[0] - (a[1] + (1 - a[0] - a[1])) / 2
    elif a.ndim == 2:
        return a[:, 0] - (a[:, 1] + (1 - a[:, 0] - a[:, 1])) / 2
    else:
        return a[:, :, 0] - (a[:, :, 1] + (1 - a[:, :, 0] - a[:, :, 1])) / 2

def kharal_score_2(a):
    """
        Calculates score of the Intuitionistic Fuzzy Set (u, v) and returns a crisp value.
        Uses a formula: (u + v) / 2 - (1 - u - v)

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Set (u, v)

        Returns
        -------
            float
                Crisp value
    """
    # cast types
    a = a.astype(float)

    if a.ndim == 1:
        return (a[0] + a[1]) / 2 - (1 - a[0] - a[1])
    elif a.ndim == 2:
        return (a[:, 0] + a[:, 1]) / 2 - (1 - a[:, 0] - a[:, 1])
    else:
        return (a[:, :, 0] + a[:, :, 1]) / 2 - (1 - a[:, :, 0] - a[:, :, 1])

def liu_wang_score(a):
    """
        Calculates score of the Intuitionistic Fuzzy Set (u, v) and returns a crisp value.
        Uses a formula: u + u * (1 - u - v)

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Set (u, v)

        Returns
        -------
            float
                Crisp value
    """
    # cast types
    a = a.astype(float)

    if a.ndim == 1:
        return a[0] + a[0] * (1 - a[0] - a[1])
    elif a.ndim == 2:
        return a[:, 0] + a[:, 0] * (1 - a[:, 0] - a[:, 1])
    else:
        return a[:, :, 0] - a[:, :, 0] * (1 - a[:, :, 0] - a[:, :, 1])

def supriya_score(a):
    """
        Calculates score of the Intuitionistic Fuzzy Set (u, v) and returns a crisp value.
        Uses a formula: (u - v * (1 - u - v))

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Set (u, v)

        Returns
        -------
            float
                Crisp value
    """
    # cast types
    a = a.astype(float)

    if a.ndim == 1:
        return a[0] - a[1] * (1 - a[0] - a[1])
    elif a.ndim == 2:
        return a[:, 0] - a[:, 1] * (1 - a[:, 0] - a[:, 1])
    else:
        return a[:, :, 0] - (1 - a[:, :, 0] - a[:, :, 1])

def thakur_score(a):
    """
        Calculates score of the Intuitionistic Fuzzy Set (u, v) and returns a crisp value.
        Uses a formula: (u**v - v**2) 

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Set (u, v)
            
        Returns
        -------
            float
                Crisp value
    """
    # cast types
    a = a.astype(float)

    if a.ndim == 1:
        return a[0]**2 - a[1]**2
    elif a.ndim == 2:
        return a[:, 0]**2 - a[:, 1]**2
    else:
        return a[:, :, 0] ** 2 - a[:, :, 1] ** 2

def wan_dong_score_1(a):
    """
        Calculates score of the Intuitionistic Fuzzy Set (u, v) and returns a crisp value.
        Uses a formula: 1/2 * ((u - v) / 2 + 1)

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Set (u, v)

        Returns
        -------
            float
                Crisp value
    """
    # cast types
    a = a.astype(float)
    
    if a.ndim == 1:
        return 1/2 * ((a[0] - a[1]) / 2 + 1)
    elif a.ndim == 2:
        return 1/2 * ((a[:, 0] - a[:, 1]) / 2 + 1)
    else:
        return 1/2 * ((a[:, :, 0] - a[:, :, 1]) / 2 + 1)

def wan_dong_score_2(a):
    """
        Calculates score of the Intuitionistic Fuzzy Set (u, v) and returns a crisp value.
        Uses a formula: ((u - v) + 1) / 2

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Set (u, v)

        Returns
        -------
            float
                Crisp value
    """
    # cast types
    a = a.astype(float)

    if a.ndim == 1:
        return ((a[0] - a[1]) + 1) / 2
    elif a.ndim == 2:
        return ((a[:, 0] - a[:, 1]) + 1) / 2
    else:
        return ((a[:, :, 0] - a[:, :, 1]) + 1) / 2

def wei_score(a):
    """
        Calculates score of the Intuitionistic Fuzzy Set (u, v) and returns a crisp value.
        Uses a formula (1 - u - v)

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Set (u, v)

        Returns
        -------
            float
                Crisp value
    """
    # cast types
    a = a.astype(float)

    if a.ndim == 1:    
        p = 1 - a[0] - a[1]
        return np.cos(np.abs(a[0] - a[1]) / (2 * (1 + p)) * np.pi)
    elif a.ndim == 2:    
        p = 1 - a[:, 0] - a[:, 1]
        return np.cos(np.abs(a[:, 0] - a[:, 1]) / (2 * (1 + p)) * np.pi)
    else:
        p = 1 - a[:, :, 0] - a[:, :, 1]
        return np.cos(np.abs(a[:, :, 0] - a[:, :, 1]) / (2 * (1 + p)) * np.pi)

def zhang_xu_score_1(a):
    """
        Calculates score of the Intuitionistic Fuzzy Set (u, v) and returns a crisp value.
        Uses a formula: ((1 - v) / (2 - u - v))

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Set (u, v)

        Returns
        -------
            float
                Crisp value
    """
    # cast types
    a = a.astype(float)
    
    if a.ndim == 1:
        return (1 - a[1]) / (2 - a[0] - a[1])
    elif a.ndim == 2:
        return (1 - a[:, 1]) / (2 - a[:, 0] - a[:, 1])
    else:
        return (1 - a[:, :, 1]) / (2 - a[:, :, 0] - a[:, :, 1])
        
def zhang_xu_score_2(a):
    """
        Calculates score of the Intuitionistic Fuzzy Set (u, v) and returns a crisp value.
        Uses a formula: (1 - (1 - u) / (1 - u - v)) 

        Parameters
        ----------
            a : ndarray
                Intuitionistic Fuzzy Set (u, v)
            
        Returns
        -------
            float
                Crisp value
    """
    # cast types
    a = a.astype(float)
    
    if a.ndim == 1:
        if 1 - (1 - a[0] - a[1]) == 0:
            return 0
        return 1 - (1 - a[0]) / (1 - (1 - a[0] - a[1]))
    elif a.ndim == 2:
        return np.nan_to_num(1 - (1 - a[:, 0]) / (1 - (1 - a[:, 0] - a[:, 1])))
    else:
        return np.nan_to_num(1 - (1 - a[:, :, 0]) / (1 - (1 - a[:, :, 0] - a[:, :, 1])))
