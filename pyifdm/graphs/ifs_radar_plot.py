# Copyright (c) 2023 Jakub Więckowski

import matplotlib.pyplot as plt
import numpy as np

def ifs_radar_plot(ifses, labels=None, title=None, ax=None):
    """
    Create a Spider/Radar Plot for Intuitionistic Fuzzy Sets (IFS).

    Parameters:
    - ifses (list): A list of IFS, where each IFS is a list of three values (membership, non-membership, uncertainty).
    - labels (list, optional): Labels for each IFS. If not provided, numerical indices will be used.
    - title (str, optional): The title for the plot.
    - ax (Axes or None): Axes object to draw on. If None, then the current axes are used.

    Returns:
        ax

    Example:
    ```
    # Example Usage:
    ifses_example = [[0.6, 0.2, 0.2], [0.8, 0.1, 0.1], [0.4, 0.3, 0.3]]
    labels_example = ['IFS 1', 'IFS 2', 'IFS 3']
    ifs_radar_plot(ifses_example, labels=labels_example, title='IFS Comparison')
    ```

    The Spider/Radar Plot visually represents the degree of belief for membership, non-membership, and uncertainty in each IFS.
    """

    if ax is None:
        ax = plt.subplot(111, polar=True)


    num_ifses = len(ifses)
    num_vars = len(ifses[0])
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is circular, so we need to close the plot loop by appending the start element to the end
    ifses += [ifses[0]]
    angles += [angles[0]]

    # Plot each IFS
    for i in range(num_ifses):
        values = ifses[i]
        values += [values[0]]
        ax.fill(angles, values, alpha=0.25, label=labels[i] if labels else f'IFS {i + 1}')

    # Add labels and title
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_yticklabels([])
    ax.set_yticks([])

    ax.set_xticks(angles[:-1], ['Membership', 'Non-membership', 'Uncertainty'])
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    ax.set_title(title if title else 'Intuitionistic Fuzzy Sets Comparison')

    return ax