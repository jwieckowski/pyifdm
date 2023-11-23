# Copyright (c) 2023 Jakub Więckowski

import numpy as np

class IFS:
    """
    Represents an Intuitionistic Fuzzy Set (IFS) with membership, non-membership, and uncertainty values.

    Parameters:
    - membership (float): The degree to which an element belongs to the set.
    - non_membership (float): The degree to which an element does not belong to the set.
    - uncertainty (float, optional): The degree of uncertainty in the membership assignment.
      If not provided, it is calculated as 1 - membership - non_membership.

    Methods:
    - __and__(self, other): Intersection operator for IFS.
    - __or__(self, other): Union operator for IFS.
    - __invert__(self): Complement operator for IFS.
    - owa_aggregation(self, weights): Ordered Weighted Averaging (OWA) aggregation operator for IFS.

    Example:
    ```
    # Example Usage:
    ifs1 = IFS(0.6, 0.2)
    ifs2 = IFS(0.8, 0.1)
    intersection_result = ifs1 & ifs2
    union_result = ifs1 | ifs2
    complement_result = ~ifs1
    owa_weights = [0.3, 0.4, 0.3]
    owa_result = ifs1.owa_aggregation(owa_weights)
    ```

    The IFS class supports standard set operations and aggregation, providing a versatile representation
    for handling uncertainty and imprecision in decision-making.
    """

    def __init__(self, membership, non_membership, uncertainty=None):
        """
        Initialize an Intuitionistic Fuzzy Set (IFS).

        Parameters:
        - membership (float): The degree to which an element belongs to the set.
        - non_membership (float): The degree to which an element does not belong to the set.
        - uncertainty (float, optional): The degree of uncertainty in the membership assignment.
            If not provided, it is calculated as 1 - membership - non_membership.
        
        """

        self.membership = membership
        self.non_membership = non_membership
        if uncertainty is not None:
            self.uncertainty = uncertainty
        else:
            self.uncertainty = 1 - membership - non_membership

    def __repr__(self):
        """
        Return a string representation of the IFS.

        Returns:
        str: A string representation of the IFS.

        """

        return f"IFS({self.membership}, {self.non_membership}, {self.uncertainty})"

    def __str__(self):
        """
        Return a human-readable string representation of the IFS.

        Returns:
        str: A string representation of the IFS.
        
        """

        return f"Membership: {self.membership}, Non-membership: {self.non_membership}, Uncertainty: {self.uncertainty}"

    def __eq__(self, other):
        """
        Check if two IFS are equal.

        Parameters:
        - other (IFS): The IFS to compare with.

        Returns:
        bool: True if the IFS are equal, False otherwise.
        
        """

        return (
            self.membership == other.membership and
            self.non_membership == other.non_membership and
            self.uncertainty == other.uncertainty
        )

    def __add__(self, other):
        """
        Compute the addition of two IFS.

        Parameters:
        - other (IFS): The IFS to add.

        Returns:
        IFS: The resulting IFS after addition.
        
        """
        
        membership = self.membership + other.membership - self.membership * other.membership
        non_membership = self.non_membership - other.non-membership
        uncertainty = 1 - (membership + non_membership)

        return IFS(membership, non_membership, uncertainty)

    def __sub__(self, other):
        """
        Compute the subtraction of two IFS.

        Parameters:
        - other (IFS): The IFS to subtract.

        Returns:
        IFS: The resulting IFS after subtraction.
        
        """
        
        if self.non_membership / other.non_membership <= ((1 - self.membership) / (1 - other.membership)):
            membership = (self.membership - other.membership) / (1 - other.membership)
            non_membership = self.non_membership / other.non-membership
            uncertainty = 1 - (membership + non_membership)
            
            return IFS(membership, non_membership, uncertainty)
        else:
            return IFS(0, 1, 0)

    def __mul__(self, other):
        """
        Compute the multiplication of two IFS.

        Parameters:
        - other (IFS): The IFS to multiply.

        Returns:
        IFS: The resulting IFS after multiplication.
        
        """
        
        membership = self.membership * other.membership
        non_membership = self.non_membership + other.non_membership - self.non_membership * other.non_membership
        uncertainty = 1 - (membership + non_membership)

        return IFS(membership, non_membership, uncertainty)

    def __truediv__(self, other):
        """
        Compute the division of two IFS.

        Parameters:
        - other (IFS): The IFS to divide.

        Returns:
        IFS: The resulting IFS after division.
        
        """
        
        if self.membership / other.membership <= ((1 - self.non_membership) / (1 - other.non_membership)):
            membership = self.membership / other.membership
            non_membership = (self.non_membership - other.non_membership) / (1 - other.non_membership)
            uncertainty = 1 - (membership + non_membership)
            
            return IFS(membership, non_membership, uncertainty)
        else:
            return IFS(1, 0, 0)

    def __pow__(self, y):
        """
        Compute the division of two IFS.

        Parameters:
        - other (IFS): The IFS to divide.

        Returns:
        IFS: The resulting IFS after division.
        
        """
        
        if y <= 0:
            raise ValueError('Power should be greater than 0')

        membership = self.membership ** y
        non_membership = 1 - (1 - self.non_membership) ** y
        uncertainty = 1 - (membership + non_membership)

        return IFS(membership, non_membership, uncertainty)

    def __and__(self, other):
        """
        Compute the intersection of two IFS.

        Parameters:
        - other (IFS): The IFS to intersect with.

        Returns:
        IFS: The resulting IFS after intersection.
        
        """

        membership = min(self.membership, other.membership)
        non_membership = max(self.non_membership, other.non_membership)
        uncertainty = 1 - (membership + non_membership)

        return IFS(membership, non_membership, uncertainty)

    def __or__(self, other):
        """
        Compute the union of two IFS.

        Parameters:
        - other (IFS): The IFS to union with.

        Returns:
        IFS: The resulting IFS after union.
        
        """
    
        membership = max(self.membership, other.membership)
        non_membership = min(self.non_membership, other.non_membership)
        uncertainty = 1 - (membership + non_membership)
        
        return IFS(membership, non_membership, uncertainty)

    def __invert__(self):
        """
        Compute the complement of the IFS.

        Returns:
        IFS: The resulting complemented IFS.
        
        """

        return IFS(self.non_membership, self.membership, self.uncertainty)

    def dominance(self, other):
        """
        Check if one IFS dominates another.

        Parameters:
        - other (IFS): The IFS to compare with.

        Returns:
        bool: True if the current IFS dominates the other, False otherwise.
        
        """

        return (
            self.membership >= other.membership and
            self.non_membership <= other.non_membership and
            self.uncertainty <= other.uncertainty
        )

    def owa_aggregation(self, weights):
        """
        Compute the Ordered Weighted Averaging (OWA) aggregation of the IFS.

        Parameters:
        - weights (list): A list of weights for each component (non-membership, uncertainty, membership).

        Returns:
        float: The result of the OWA aggregation.
        
        """

        values = [self.non_membership, self.uncertainty, self.membership]
        return np.sum(np.multiply(weights, values))

    def similarity_jaccard(self, other):
        """
        Compute the Jaccard similarity coefficient between two IFS.

        Parameters:
        - other (IFS): The IFS to compare with.

        Returns:
        float: Jaccard similarity coefficient.
        """

        intersection = min(self.membership, other.membership) + min(self.non_membership, other.non_membership) + min(self.uncertainty, other.uncertainty)
        union = max(self.membership, other.membership) + max(self.non_membership, other.non_membership) + max(self.uncertainty, other.uncertainty)
        similarity = intersection / union
        
        return similarity

    def fuzzy_relation(self, other):
        """
        Compute the fuzzy relation matrix between two IFS.

        Parameters:
        - other (IFS): The IFS to form a relation with.

        Returns:
        np.ndarray: Fuzzy relation matrix.
        
        """

        relation_matrix = np.zeros((3, 3))
        for i, component_i in enumerate([self.membership, self.non_membership, self.uncertainty]):
            for j, component_j in enumerate([other.membership, other.non_membership, other.uncertainty]):
                relation_matrix[i, j] = min(component_i, component_j)
        return relation_matrix
