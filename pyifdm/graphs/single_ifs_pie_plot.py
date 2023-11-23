# Copyright (c) 2023 Jakub Więckowski

import matplotlib.pyplot as plt
import numpy as np

def single_ifs_pie_plot(ifs, title=None, ax=None):
    """
    Visualize a single Intuitionistic Fuzzy Set (IFS) using a pie plot.

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
    single_ifs_pie_plot(ifs_example, title='Example IFS')
    ```

    The pie plot displays the degree of belief for membership, non-membership, and uncertainty of an Intuitionistic Fuzzy Set.
    """

    categories = ['Membership', 'Non-membership', 'Uncertainty']
    values = np.array(ifs)

    if ax is None:
        ax = plt.gca()

    # Create a pie plot
    colors = ['dodgerblue', 'red', 'gray']
    ax.pie(values, labels=categories, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops=dict(edgecolor='black'))

    # Add title
    ax.set_title(title if title else 'Intuitionistic Fuzzy Set')

    return ax
