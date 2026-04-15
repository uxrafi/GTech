"""
Random Tree Learner (RTLearner)
Builds decision trees by randomly selecting features to split on (using median values),
creating a model that predicts continuous values through hierarchical if-then rules.
"""

# Implementation Steps:
# STEP 1: Initialize learner with leaf_size and verbosity settings
# STEP 2: Build tree recursively by selecting random features and median splits
# STEP 3: Handle base cases (leaf nodes) when data is homogeneous or too small
# STEP 4: Store trained tree structure as numpy array
# STEP 5: Make predictions by traversing the tree for each query point
# STEP 6: Support both single-point and batch queries

# Numerical computing package for array operations
import numpy as np
# Random number generation for feature selection
import random

class RTLearner:
    # STEP 1: Initialize learner with leaf_size and verbosity settings
    def __init__(self, leaf_size=1, verbose=False):
        """Initialize RTLearner with specified leaf size and verbosity."""
        self.leaf_size = leaf_size  # Minimum samples per leaf
        self.verbose = verbose      # Debug output flag
        self.tree = None            # Will store the decision tree

    def author(self):
        """Return author's GT username for grading purposes."""
        return "urafi3"

    def study_group(self):
        """Return study group members as comma-separated string."""
        return "urafi3"

    
    def build_tree(self, X, y):
        """
        Recursively builds random decision tree.
        Tree structure: [feature_index, split_value, left_node_offset, right_node_offset]
        Leaf nodes: ["leaf", predicted_value, np.nan, np.nan]
        """

        # STEP 3: Handle base cases (leaf nodes) when data is homogeneous or too small
        # Base case 1: Fewer samples than leaf size
        # Base case 2: All y values are identical
        if len(X) <= self.leaf_size or len(np.unique(y)) == 1:
            return np.array([["leaf", np.mean(y), np.nan, np.nan]])
        
        # STEP 2: Build tree recursively by selecting random features and median splits
        # Randomly select feature and split at median
        feat = random.randint(0, X.shape[1]-1)
        split_val = np.median(X[:, feat])
        
        # Split data
        left = X[:, feat] <= split_val
        
        # Handle case where split doesn't divide data
        if np.all(left) or np.all(~left):
            return np.array([["leaf", np.mean(y), np.nan, np.nan]])
        
        # Recursively build left and right subtrees
        left_tree = self.build_tree(X[left], y[left])
        right_tree = self.build_tree(X[~left], y[~left])
        
        # Create current node with pointers to children
        root = np.array([[feat, split_val, 1, left_tree.shape[0]+1]])
        return np.vstack((root, left_tree, right_tree))

    # STEP 4: Store trained tree structure as numpy array
    def add_evidence(self, data_x, data_y):
        """Train the model on provided data."""
        self.tree = self.build_tree(data_x, data_y)

    # STEP 6: Support both single-point and batch queries
    def query(self, points):
        """Make predictions for multiple input points."""
        return np.array([self._query_one(point) for point in points])
    

    # STEP 5: Make predictions by traversing the tree for each query point
    def _query_one(self, point, node=0):
        """
        Internal method for single-point prediction.
        Traverses tree recursively until leaf node is found.
        """
        node_data = self.tree[node]
        
        # If leaf node, return prediction
        if node_data[0] == "leaf":
            return float(node_data[1])
        
        # Get feature index and split value
        feat = int(float(node_data[0]))
        
        # Traverse left or right subtree
        if point[feat] <= float(node_data[1]):
            return self._query_one(point, node + int(float(node_data[2])))
        return self._query_one(point, node + int(float(node_data[3])))