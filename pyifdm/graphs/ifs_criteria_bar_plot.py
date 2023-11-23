# Copyright (c) 2023 Jakub Więckowski

import matplotlib.pyplot as plt
import numpy as np

def ifs_criteria_bar_plot(criteria_weights, criteria_names=None, ax=None):
    """
    Visualize a list of Intuitionistic Fuzzy Sets (IFS) as stacked bars.

    Parameters:
    - criteria_weights (list): A list of lists, where each inner list represents an IFS (membership, non-membership, uncertainty).
    - criteria_names (list, optional): A list of names for each criteria. If not provided, default names will be used.
    - ax (Axes or None): Axes object to draw on. If None, then the current axes are used.

    Returns:
        ax

    Example:
    ```
    # Example Usage:
    criteria_weights_example = [[0.6, 0.2, 0.2], [0.8, 0.1, 0.1], [0.5, 0.3, 0.2]]
    criteria_names_example = ['Criterion 1', 'Criterion 2', 'Criterion 3']
    ifs_criteria_bar_plot(criteria_weights_example, criteria_names_example)
    ```

    The stacked bar plot displays the degree of belief for membership, non-membership, and uncertainty of multiple criteria.
    """

    num_criteria = len(criteria_weights)
    num_components = len(criteria_weights[0])

    if criteria_names is None:
        criteria_names = [f'$C_{{{i + 1}}}$' for i in range(num_criteria)]

    if ax is None:
        ax = plt.gca()

    # colors
    colors = ['dodgerblue', 'red', 'gray']

    # Create stacked bar plot
    bottom_values = np.zeros(num_components)
    for i in range(num_criteria):
        values = np.array(criteria_weights[i])
        bottom_values = 0
        for idx, val in enumerate(values):
            ax.bar(criteria_names[i], val, bottom=bottom_values, alpha=0.7, color=colors[idx])
            bottom_values += val

    # Add labels and title
    ax.set_ylabel('Degree of Belief')

    # Add legend
    ax.legend(['Membership', 'Non-membership', 'Uncertainty'], bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)

    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Adjust ylim
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    return ax
