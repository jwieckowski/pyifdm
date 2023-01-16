# # Copyright (c) 2022 Jakub Więckowski
# # Copyright (c) 2022 Bartłomiej Kizielewicz

import numpy as np

__all__ = [
    'burillo_entropy_weights',
    'equal_weights',
    'entropy_weights',
    'liu_entropy_weights',
    'szmidt_entropy_weights',
    'thakur_entropy_weights',
    'ye_entropy_weights',
]

def burillo_entropy_weights(matrix):
    """
        Calculates the objective weights for Intuitionistic Fuzzy Matrix, weight depend on the Burillo entropy measure in the column

        Parameters
        ----------
            matrix: ndarray
                Decision matrix / alternatives data
                Alternatives are in rows and Criteria are in columns

        Returns
        -------
            ndarray
                Array of weights based on matrix entropy
    """

    weights = np.zeros(matrix.shape[1])
    for i in range(matrix.shape[1]):
        weights[i] = np.sum(1 - (matrix[:, i, 0] + matrix[:, i, 1])) / matrix[:, i].shape[0]
    
    return (1 - weights) / np.sum(1 - weights)

def equal_weights(matrix):
    """
        Calculates the objective weights for Intuitionistic Fuzzy Matrix, each weight will have the same value

        Parameters
        ----------
            matrix: ndarray
                Decision matrix / alternatives data
                Alternatives are in rows and Criteria are in columns

        Returns
        -------
            ndarray
                Array of equal weights
    """

    w = np.ones(matrix.shape[1]) / 2
    return np.repeat(w, 2).reshape((len(w), 2))


def entropy_weights(matrix):
    """
        Calculates the objective weights for Intuitionistic Fuzzy Matrix, weight depend on the entropy measure in the column

        Parameters
        ----------
            matrix: ndarray
                Decision matrix / alternatives data
                Alternatives are in rows and Criteria are in columns

        Returns
        -------
            ndarray
                Array of weights based on matrix entropy
    """

    p = []
    for j in range(matrix.shape[1]):
        p.append(1 / matrix.shape[0] * sum([1 - matrix[i, j, 0] - matrix[i, j, 1]
                                            for i in range(matrix.shape[0])]))

    w = np.zeros((1, matrix.shape[1]))
    for idx, pp in enumerate(p):
        w[0, idx] = (1/pp) / np.sum(1/np.array(p))

    return w

def liu_entropy_weights(matrix):
    """
        Calculates the objective weights for Intuitionistic Fuzzy Matrix, weight depend on the Liu entropy measure in the column

        Parameters
        ----------
            matrix: ndarray
                Decision matrix / alternatives data
                Alternatives are in rows and Criteria are in columns

        Returns
        -------
            ndarray
                Array of weights based on matrix entropy
    """
    
    weights = np.zeros(matrix.shape[1])
    for i in range(matrix.shape[1]):
        value = (np.pi / 4) + np.abs(matrix[:, i, 0] ** 2 - matrix[:, i, 1] ** 2) / 4 * np.pi
        weights[i] = np.sum(np.cos(value) / np.sin(value)) / matrix[:, i].shape[0]
    
    return weights

def szmidt_entropy_weights(matrix):
    """
        Calculates the objective weights for Intuitionistic Fuzzy Matrix, weight depend on the Szmidt entropy measure in the column

        Parameters
        ----------
            matrix: ndarray
                Decision matrix / alternatives data
                Alternatives are in rows and Criteria are in columns

        Returns
        -------
            ndarray
                Array of weights based on matrix entropy
    """
    
    weights = np.zeros(matrix.shape[1])
    for i in range(matrix.shape[1]):
        pi = 1 - matrix[:, i, 0] - matrix[:, i, 1]
        weights[i] = np.sum((np.min(matrix[:, i]) + pi) / (np.max(matrix[:, i]) + pi)) / matrix[:, i].shape[0]
    
    return weights

def thakur_entropy_weights(matrix):
    """
        Calculates the objective weights for Intuitionistic Fuzzy Matrix, weight depend on the Thakur entropy measure in the column

        Parameters
        ----------
            matrix: ndarray
                Decision matrix / alternatives data
                Alternatives are in rows and Criteria are in columns

        Returns
        -------
            ndarray
                Array of weights based on matrix entropy
    """

    weights = np.zeros(matrix.shape[1])
    for i in range(matrix.shape[1]):
        weights[i] = np.sum((1 / np.cos(np.abs(np.abs(3 - 2 * matrix[:, i, 0] - 7 / 3) - 7 / 3) * np.pi / 7) + 1 / np.cos(
        np.abs(np.abs(3 - 2 * matrix[:, i, 1] - 7 / 3) - 7 / 3) * np.pi / 7) - 334 / 135) / (206 / 135)) / matrix[:, i].shape[0]

    return (1 - weights) / np.sum(1 - weights)


def ye_entropy_weights(matrix):
    """
        Calculates the objective weights for Intuitionistic Fuzzy Matrix, weight depend on the Ye entropy measure in the column

        Parameters
        ----------
            matrix: ndarray
                Decision matrix / alternatives data
                Alternatives are in rows and Criteria are in columns

        Returns
        -------
            ndarray
                Array of weights based on matrix entropy
    """

    weights = np.zeros(matrix.shape[1])
    for i in range(matrix.shape[1]):
        weights[i] = np.sum((np.sin((1 + matrix[:, i, 0] - matrix[:, i, 1]) * np.pi / 4) + np.sin((1 - matrix[:, i, 0] + matrix[:, i, 1]) * np.pi / 4) - 1) * (
            1 / (np.sqrt(2) - 1))) / matrix[:, i].shape[0]

    return weights
