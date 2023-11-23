# Copyright (c) 2023 Jakub Więckowski

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def ifs_heatmap_plot(ifs_matrix, cmap='Blues', labels=None, ax=None):
    """
    Visualize a matrix plot for Intuitionistic Fuzzy Sets (IFS).

    Parameters:
    - ifs_matrix (list of lists): A matrix of IFS values where each row represents an IFS.
    - labels (list of str, optional): labels for each row in the matrix.
    - ax (Axes or None): Axes object to draw on. If None, then the current axes are used.

    Returns:
        ax

    Example:
    ```
    # Example Usage:
    ifs_matrix_example = [
        [0.6, 0.2, 0.2],
        [0.8, 0.1, 0.1],
        [0.4, 0.3, 0.3]
    ]
    labels_example = ['Set 1', 'Set 2', 'Set 3']
    ifs_heatmap_plot(ifs_matrix_example, labels=labels_example)
    ```

    The matrix plot displays the degree of belief for membership, non-membership, and uncertainty of multiple Intuitionistic Fuzzy Sets.
    """

    # Convert the ifs_matrix to a NumPy array for compatibility
    ifs_matrix = np.array(ifs_matrix)

    if ax is None:
        ax = plt.gca()

    # Create matrix plot
    im = ax.imshow(ifs_matrix, cmap=cmap, vmin=0, vmax=1)

    # Normalize the threshold to the images color range.
    threshold = im.norm(ifs_matrix.max())/2.

    # Set labels and title
    if labels is None:
        labels = [f'IFS {i+1}' for i in range(len(ifs_matrix))]

    ax.set_yticks(np.arange(len(labels)) if labels else np.arange(len(ifs_matrix)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(['Membership', 'Non-membership', 'Uncertainty'])
    ax.set_ylabel('Intuitionistic Fuzzy Sets')

    kw = dict(horizontalalignment="center", verticalalignment="center")
    textcolors=("black", "white")

    # Get the formatter in case a string is supplied
    valfmt = matplotlib.ticker.StrMethodFormatter("{x:.2f}")

    # Loop over data dimensions and create text annotations.
    for i in range(len(ifs_matrix)):
        for j in range(len(ifs_matrix)):
            kw.update(color=textcolors[int(im.norm(ifs_matrix[i, j]) > threshold)])
            im.axes.text(j, i, valfmt(ifs_matrix[i, j], None), **kw)

    # Display colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Degree of Belief')

    return ax