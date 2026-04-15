"""
DTLearner.py - Decision Tree Learner Implementation

This module implements a decision tree regression/classification algorithm.
The tree uses correlation-based feature splitting and median values to create
a binary tree structure for making predictions.It follows the following algorithm (JR Quinlan):

build_tree(data)
    if data.shape[0] == 1: return [leaf, data.y, NA, NA]
    if all data.y same: return [leaf, data.y, NA, NA]
    else
        determine best feature i to split on
        SplitVal = data[:,i].median()
        lefttree = build_tree(data[data[:,i]<=SplitVal])
        righttree = build_tree(data[data[:,i]>SplitVal])
        root = [i, SplitVal, 1, lefttree.shape[0] + 1]
        return (append(root, lefttree, righttree))
"""

# STEP 1: Initialize the decision tree parameters
# STEP 2: Select the best feature to split on
# STEP 3: Split the data at the median value
# STEP 4: Recursively build left and right subtrees
# STEP 5: Make predictions by traversing the tree

import numpy as np    # Import numpy for numerical operations

class DTLearner(object):
    """
    Decision Tree Learner for regression and classification tasks.
    """

    # STEP 1: Initialize the decision tree parameters 
    def __init__(self, leaf_size=1, verbose=False):
        """Initialize with leaf_size and verbosity settings"""
        self.leaf_size = max(1, leaf_size)  # Ensure minimum leaf_size of 1
        self.tree = None  # Will store the decision tree structure
        self.verbose = verbose

    def author(self):
        """Return GT username for grading"""
        return "urafi3"

    def study_group(self):
        """Return study group member name"""
        return "urafi3"

    def add_evidence(self, data_x, data_y):
        """Build the decision tree using training data"""
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        """Recursively builds the decision tree"""
        # Base case 1: If number of samples is <= leaf_size, return a leaf node
        if data_x.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])
        
        # Base case 2: If all y values are the same (with tolerance), return leaf
        if np.all(np.abs(data_y - data_y.mean()) < 1e-6):
            return np.array([[-1, data_y[0], np.nan, np.nan]])
        
        # STEP 2: Select the best feature to split on
        with np.errstate(invalid='ignore'):  # Ignore invalid correlation warnings
            corr = np.array([np.abs(np.corrcoef(data_x[:,i], data_y)[0,1]) 
                           if (np.std(data_x[:,i]) > 1e-6 and np.std(data_y) > 1e-6)
                           else -1 
                           for i in range(data_x.shape[1])])
        best_feat = np.nanargmax(corr)  # Get index of feature with max correlation

        # STEP 3: Split the data at the median value with small noise
        split_val = np.median(data_x[:, best_feat])
        split_val += np.random.normal(0, 1e-3 * np.std(data_x[:, best_feat]))
        
        # Create masks for left and right branches
        left_mask = data_x[:, best_feat] <= split_val
        
        # Base case 3: If split doesn't properly divide data, return leaf
        min_samples = max(2, self.leaf_size//2)  # Ensure minimum samples per split
        if sum(left_mask) < min_samples or sum(~left_mask) < min_samples:
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])
        
        # STEP 4: Recursively build left and right subtrees
        left_tree = self.build_tree(data_x[left_mask], data_y[left_mask])
        right_tree = self.build_tree(data_x[~left_mask], data_y[~left_mask])
        
        root = np.array([[best_feat, split_val, 1, left_tree.shape[0] + 1]])
        
        # Combine root with left and right subtrees
        return np.vstack((root, left_tree, right_tree))

    # STEP 5: Make predictions by traversing the tree
    def query(self, points):
        """Make predictions for input points using the built tree"""
        results = np.zeros(points.shape[0])
        for i in range(points.shape[0]):
            node = 0  # Start at root node
            while True:
                feat, split, left, right = self.tree[node]
                if feat == -1:  # Leaf node
                    results[i] = split
                    break
                if points[i, int(feat)] <= split:
                    node += int(left)  # Move to left child
                else:
                    node += int(right)  # Move to right child
        return results

if __name__ == "__main__":
    print("DTLearner implementation ready for testing")