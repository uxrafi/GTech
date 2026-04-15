"""
Random Tree Learner (RTLearner)
Builds decision trees by randomly selecting features to split on (using median values),
creating a model that predicts continuous values through hierarchical if-then rules.

Updated to support classification tasks by returning the mode of labels in leaves
instead of the mean.
"""

import numpy as np
import random
from scipy.stats import mode  # For mode calculation in classification leaves

class RTLearner:
    def __init__(self, leaf_size=5, verbose=False):
        """
        Initialize RTLearner with specified leaf size and verbosity.
        
        Parameters:
        leaf_size (int): Minimum samples per leaf node (>=5 recommended for classification)
        verbose (bool): Enable debug prints if True
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None  # Will store the tree as a numpy array

    def author(self):
        """Return author's GT username for grading purposes."""
        return "urafi3"

    def study_group(self):
        """Return study group members as comma-separated string."""
        return "urafi3"

    def build_tree(self, X, y):
        """
        Recursively builds random decision tree.
        
        Tree structure:
        Each node is an array: [feature_index, split_value, left_node_offset, right_node_offset]
        Leaf nodes: ["leaf", predicted_class, np.nan, np.nan]

        Parameters:
        X (np.ndarray): Feature data, shape (n_samples, n_features)
        y (np.ndarray): Target labels, shape (n_samples,)

        Returns:
        np.ndarray: The subtree rooted at the current call
        """
        # STEP 3: Base cases for leaf nodes
        # 1) Number of samples <= leaf_size
        # 2) All labels identical
        if len(X) <= self.leaf_size or len(np.unique(y)) == 1:
            # For classification, leaf prediction is mode of y (most common class)
            pred_class = mode(y).mode[0]
            return np.array([["leaf", pred_class, np.nan, np.nan]])

        # STEP 2: Select random feature and split at median
        feat = random.randint(0, X.shape[1] - 1)
        split_val = np.median(X[:, feat])

        # Split data according to median split
        left_mask = X[:, feat] <= split_val

        # Handle edge case where split does not partition data
        if np.all(left_mask) or np.all(~left_mask):
            pred_class = mode(y).mode[0]
            return np.array([["leaf", pred_class, np.nan, np.nan]])

        # Recursively build left and right subtrees
        left_tree = self.build_tree(X[left_mask], y[left_mask])
        right_tree = self.build_tree(X[~left_mask], y[~left_mask])

        # Current node: [feature, split_value, left_offset, right_offset]
        # left_offset is always 1 (next row), right_offset is size of left_tree + 1
        root = np.array([[feat, split_val, 1, left_tree.shape[0] + 1]])

        # Stack current node + left subtree + right subtree vertically
        return np.vstack((root, left_tree, right_tree))

    def add_evidence(self, data_x, data_y):
        """
        Train the learner by building the decision tree.

        Parameters:
        data_x (np.ndarray): Training features
        data_y (np.ndarray): Training labels
        """
        self.tree = self.build_tree(data_x, data_y)
        if self.verbose:
            print("Tree built with {} nodes".format(self.tree.shape[0]))

    def query(self, points):
        """
        Predict labels for given query points.

        Parameters:
        points (np.ndarray): Query features, shape (n_queries, n_features)

        Returns:
        np.ndarray: Predicted labels for each query point
        """
        return np.array([self._query_one(point) for point in points])

    def _query_one(self, point, node=0):
        """
        Predict label for a single data point by traversing the tree.

        Parameters:
        point (np.ndarray): Feature vector of a single query
        node (int): Current node index in the tree

        Returns:
        predicted class label
        """
        node_data = self.tree[node]

        # Check if leaf node
        if node_data[0] == "leaf":
            return node_data[1]  # Predicted class

        # Internal node: parse feature index and split value
        feat = int(float(node_data[0]))
        split_val = float(node_data[1])

        # Decide whether to traverse left or right subtree
        if point[feat] <= split_val:
            return self._query_one(point, node + int(float(node_data[2])))
        else:
            return self._query_one(point, node + int(float(node_data[3])))
