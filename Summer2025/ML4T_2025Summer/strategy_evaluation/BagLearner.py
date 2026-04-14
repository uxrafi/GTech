"""
Bagging Learner (BagLearner)
Implements bootstrap aggregating (bagging) to create an ensemble of learners,
improving prediction accuracy and reducing variance.

This version is updated to support **classification tasks**, aggregating
the outputs from base learners using majority voting (mode).
"""

import numpy as np
from scipy.stats import mode  # For majority vote in classification

class BagLearner(object):
    def __init__(self, learner, kwargs={"argument1": 1, "argument2": 2}, bags=20, boost=False, verbose=False):
        """
        Bootstrap Aggregation Learner (BagLearner) constructor.

        Parameters:
        learner (class): The learner class to use (must implement add_evidence/query)
        kwargs (dict): Arguments to pass to each learner's constructor
        bags (int): Number of learners in the ensemble
        boost (bool): [Optional] Whether to implement boosting (not used here)
        verbose (bool): Whether to print debugging information
        """
        # Initialize a list of learners
        self.learners = [learner(**kwargs) for _ in range(bags)]
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

    def author(self):
        """Return GT username for grading purposes."""
        return "urafi3"

    def study_group(self):
        """Return study group members as comma-separated string."""
        return "urafi3"

    def add_evidence(self, data_x, data_y):
        """
        Train each learner on a bootstrap sample of the data.

        Parameters:
        data_x (np.ndarray): Features (n_samples, n_features)
        data_y (np.ndarray): Labels or targets (n_samples,)
        """
        n_samples = data_x.shape[0]
        for learner in self.learners:
            # STEP 2: Create bootstrap sample (sampling with replacement)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = data_x[indices]
            y_bootstrap = data_y[indices]
            
            # STEP 3: Train learner on the bootstrap sample
            learner.add_evidence(X_bootstrap, y_bootstrap)
            
            if self.verbose:
                print(f"Trained learner on {len(indices)} samples")

    def query(self, points):
        """
        Make predictions by aggregating predictions from all learners using majority voting.

        Parameters:
        points (np.ndarray): Query points (n_queries, n_features)

        Returns:
        np.ndarray: Aggregated class predictions (n_queries,)
        """
        # STEP 4: Get predictions from all learners
        predictions = np.array([learner.query(points) for learner in self.learners])  # shape: (bags, n_queries)
        
        # STEP 5: For classification - aggregate using majority vote (mode)
        # mode returns both mode value and count; we need the mode value
        # Removed 'keepdims' for compatibility with older scipy versions
        mode_preds, _ = mode(predictions, axis=0)
        
        return mode_preds.flatten()  # Return a 1D array of predictions
