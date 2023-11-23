# Copyright (c) 2023 Jakub Więckowski

import matplotlib.pyplot as plt
import numpy as np

def single_ifs_bar_plot(ifs, title=None, ax=None):
    """
    Visualize a single Intuitionistic Fuzzy Set (IFS) using a bar plot.

    Parameters:
    - ifs (list): A list of three values representing the membership, non-membership, and uncertainty of the IFS.
    - title (str, optional): The title for the plot.
    - ax (Axes or None): Axes object to draw on. If None, then current axes is used.

    Returns:
        ax

    Example:
    ```
    # Example Usage:
    ifs_example = [0.6, 0.2, 0.2]
    single_ifs_bar_plot(ifs_example, title='Example IFS')
    ```

    The bar plot displays the degree of belief for membership, non-membership, and uncertainty of an Intuitionistic Fuzzy Set.
    """

    categories = ['Membership', 'Non-membership', 'Uncertainty']
    values = np.array(ifs)

    if ax is None:
        ax = plt.gca()

    # Create a bar plot with grid and edge color
    ax.bar(categories, values, color=['dodgerblue', 'red', 'gray'], edgecolor='black', alpha=0.7)

    # Add values on top of the bars
    for i, value in enumerate(values):
        ax.text(i, value + 0.01, f'{value:.2f}', ha='center', va='bottom')

    # Add labels and title
    ax.set_ylabel('Degree of Belief')
    ax.set_title(title if title else 'Intuitionistic Fuzzy Set')

    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # adjust ylim
    ax.set_ylim([0, max(ifs)+0.1*max(ifs)])

    return ax

