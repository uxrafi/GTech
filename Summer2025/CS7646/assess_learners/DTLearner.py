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
        self.leaf_size = leaf_size  # Minimum number of samples allowed in a leaf
        self.tree = None  # Will store the decision tree structure

    def author(self):
        return "urafi3"  # Returns author identifier

    def study_group(self):
        return "urafi3"  # Returns study group member name

    def add_evidence(self, data_x, data_y):
        """Build the decision tree using training data"""
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        """Recursively builds the decision tree"""
        # Base case 1: If number of samples is <= leaf_size, return a leaf node
        if data_x.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])  # Leaf node format: [-1, prediction, NA, NA]
        
        # Base case 2: If all y values are the same, return a leaf node
        if np.all(data_y == data_y[0]):
            return np.array([[-1, data_y[0], np.nan, np.nan]])
        
        # STEP 2: Select the best feature to split on
        corr = np.array([np.abs(np.corrcoef(data_x[:,i], data_y)[0,1]) 
                        if np.std(data_x[:,i]) != 0 else -1 
                        for i in range(data_x.shape[1])])
        best_feat = np.nanargmax(corr)  # Get index of feature with max correlation
        

        # STEP 3: Split the data at the median value
        split_val = np.median(data_x[:, best_feat])
        
        # Create masks for left and right branches
        left_mask = data_x[:, best_feat] <= split_val
        
        # Base case 3: If split doesn't divide the data, return leaf
        if np.all(left_mask) or np.all(~left_mask):
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
        results = []
        for point in points:
            node = 0  # Start at root node (index 0)
            while True:
                node_data = self.tree[node]
                if node_data[0] == -1:  # If leaf node
                    results.append(node_data[1])  # Use leaf's prediction
                    break
                # Decide whether to go left or right
                if point[int(node_data[0])] <= node_data[1]:
                    node += int(node_data[2])  # Move to left child
                else:
                    node += int(node_data[3])  # Move to right child
        return np.array(results)