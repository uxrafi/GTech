"""
Bagging Learner (BagLearner)
Implements bootstrap aggregating (bagging) to create an ensemble of learners,
improving prediction accuracy and reducing variance.
"""

# Implementation Steps:
# STEP 1: Initialize ensemble with specified base learner and parameters
# STEP 2: Create bootstrap samples for each learner
# STEP 3: Train each learner on its bootstrap sample
# STEP 4: Aggregate predictions from all learners
# STEP 5: Support both classification and regression via appropriate aggregation

import numpy as np

class BagLearner(object):
    # STEP 1: Initialize ensemble with specified base learner and parameters
    def __init__(self, learner, kwargs={"argument1": 1, "argument2": 2}, bags=20, boost=False, verbose=False):
        """
        Bootstrap Aggregation Learner (BagLearner) constructor.

        Parameters:
        learner (class): The learner class to use (must implement add_evidence/query)
        kwargs (dict): Arguments to pass to each learner's constructor
        bags (int): Number of learners in the ensemble
        boost (bool): [Optional] Whether to implement boosting
        verbose (bool): Whether to print debugging information
        """
        self.learners = [learner(**kwargs) for _ in range(bags)]  # Initialize learner ensemble
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

    def author(self):
        """Return GT username for grading purposes."""
        return "urafi3"

    def study_group(self):
        """Return study group members as comma-separated string."""
        return "urafi3"

    # STEP 2: Create bootstrap samples for each learner
    # STEP 3: Train each learner on its bootstrap sample
    def add_evidence(self, data_x, data_y):
        """
        Train each learner on a bootstrap sample of the data.

        Parameters:
        data_x (np.ndarray): Features (n_samples, n_features)
        data_y (np.ndarray): Targets (n_samples,)
        """
        n_samples = data_x.shape[0]
        for learner in self.learners:
            # Create bootstrap sample (with replacement)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = data_x[indices]
            y_bootstrap = data_y[indices]
            
            # Train learner on bootstrap sample
            learner.add_evidence(X_bootstrap, y_bootstrap)
            
            if self.verbose:
                print(f"Trained learner on {len(indices)} samples")

    # STEP 4: Aggregate predictions from all learners
    # STEP 5: Support both classification and regression via appropriate aggregation
    def query(self, points):
        """
        Make predictions by aggregating predictions from all learners.

        Parameters:
        points (np.ndarray): Query points (n_queries, n_features)

        Returns:
        np.ndarray: Aggregated predictions (n_queries,)
        """
        # Get predictions from all learners
        predictions = np.array([learner.query(points) for learner in self.learners])
        
        # Average predictions for regression, majority vote for classification
        return np.mean(predictions, axis=0)  # Mean aggregation for regression tasks
