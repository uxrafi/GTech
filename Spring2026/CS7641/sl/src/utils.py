"""
Utility functions used across the projet.

What this does:
- Evaluates model performance with apropriate metrics (binary vs multiclass)
- Measures training and prediction times for benchmarking
- Sets random seeds across numpy and PyTorch for reproducability

These are helper functions used by the main analisys script to keep code clean.
"""

##########################

"""
Assignment Requirements Covered:

- Evaluation metrics implementation:
  * Binary: accuracy, F1, PR-AUC (when probabilities available)
  * Multiclass: accuracy, macro-F1
- Runtime measurement: wall-clock fit and predict times
- Random seed management: numpy, PyTorch (CPU and GPU)
- Supports "Required evaluation metrics" and "Runtime table" requirements
- Supports "Methodology & reproducability" via seed control
"""

import time  # for measuring wall-clock execution times
import numpy as np  # array operations and random seed setting
import pandas as pd  # data manipulation (imported but not actualy used here)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc  # evaluation metrix
import torch  # PyTorch for neural nets and GPU seed setting


# Compute evaluation metrics depending on task type
# Binary: accuracy, F1, and PR-AUC (if probabilities provided)
# Multiclass: accuracy and macro-averaged F1
def evaluate_model(y_true, y_pred, y_prob=None, task='binary'):
    if task == 'binary':
        acc = accuracy_score(y_true, y_pred)  # proportion correct
        f1 = f1_score(y_true, y_pred)  # harmonic mean of precision/recall
        if y_prob is not None:
            # Calculate precision-recall curve and area under it
            precision, recall, _ = precision_recall_curve(y_true, y_prob[:,1])
            pr_auc = auc(recall, precision)  # better than ROC-AUC for imbalanced data
            return acc, f1, pr_auc
        return acc, f1
    else:  # multiclass
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')  # average F1 across all classes
        return acc, f1_macro


# Measure fit and predict time using wall-clock time
# Usefull for comparing model computational costs
def measure_time(estimator, X_train, y_train, X_test):
    start = time.time()
    estimator.fit(X_train, y_train)  # train the model
    fit_time = time.time() - start
    
    start = time.time()
    y_pred = estimator.predict(X_test)  # make predictions
    predict_time = time.time() - start
    
    return fit_time, predict_time, y_pred

# Set random seeds for reproducability across all libraries
# Ensures experiments give same results every time
def set_seeds(seed):
    np.random.seed(seed)  # numpy random operations
    torch.manual_seed(seed)  # PyTorch CPU operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch GPU operations if availble