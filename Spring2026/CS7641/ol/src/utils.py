"""
Utility functions used across the project.

What this does:

- Provides shared evaluation metrics for both binary and multiclass tasks
- Measures wall clock time for sklearn model fitting and predicting
- Sets random seeds across numpy, pytorch, and cuda for reproducability
- Computes median and IQR for aggregating results across multiple seeds

These functions are called by all four parts of the OL report to keep
metric computaion and seed managment consistant across experiments.
"""

##########################

"""
Assignment Requirements Covered:

- evaluate_model handles both binary (Adult) and multiclass (Wine) tasks
- Binary: returns accuracy and F1 — F1 is primary metric given class imbalance
- Multiclass: returns accuracy and macro F1 — macro required for Wine per assignment
- PR-AUC optionally computed for binary when probabilty scores are provided
- set_seeds fixes numpy, pytorch, and cuda randomness for deterministc results
- median_iqr supports stabillity reporting across 3 seeds as required by all parts
- measure_time supports wall clock comparision required in Part 2 compute accounting
"""

import time  # wall clock timing for measure_time
import numpy as np  # array ops and percentile calculations
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc  # evaluation metrics
import torch  # pytorch seed setting


# Compute evaluation metrics for a set of predictions.
# Handles both binary classifcation (Adult) and multiclass (Wine) depending on task arg.
# For binary: returns accuracy and F1 — optionally PR-AUC if probabilty scores provided
# For multiclass: returns accuracy and macro F1 — macro averages F1 equally across all classes
# regardless of class size, which is important for imbalanced Wine quality distribution.
def evaluate_model(y_true, y_pred, y_prob=None, task='binary'):
    if task == 'binary':
        acc = accuracy_score(y_true, y_pred)  # overall correct predictions
        f1  = f1_score(y_true, y_pred)         # balances precision and recall for imbalanced Adult data
        if y_prob is not None:
            # PR-AUC only computed when probabilty scores are available — used in SL report
            precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
            pr_auc = auc(recall, precision)
            return acc, f1, pr_auc
        return acc, f1
    else:  # multiclass — Wine Quality
        acc      = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')  # macro gives equal weight to all 7 quality classes
        return acc, f1_macro


# Measure how long an sklearn estimator takes to fit and predict.
# Returns fit time, predict time, and predictions seperately so caller
# can use whichever they need — used in SL report runtime comparisions.
def measure_time(estimator, X_train, y_train, X_test):
    start    = time.time()
    estimator.fit(X_train, y_train)  # train the model
    fit_time = time.time() - start

    start        = time.time()
    y_pred       = estimator.predict(X_test)  # generate predictions
    predict_time = time.time() - start

    return fit_time, predict_time, y_pred


# Fix random seeds across all libraries that use randomness in this project.
# Called at the start of every seed loop in Parts 1-4 to ensure results
# are fully reproducable — same seed always gives same results.
# Covers numpy (sklearn internals), pytorch (model init and training), and
# cuda (GPU randomness) if a GPU is avalible.
def set_seeds(seed):
    np.random.seed(seed)           # fixes sklearn and numpy randomness
    torch.manual_seed(seed)        # fixes pytorch CPU randomness
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # fixes pytorch GPU randomness if running on cuda


# Compute median and interquartile range (IQR = Q3 - Q1) over an array of values.
# Used to aggregate test F1 scores across 3 seeds in all parts of the OL report.
# Median is prefered over mean because it is robust to outlier seeds —
# if one seed gives an unusualy good or bad result it wont skew the median.
# IQR measures how spread out the results are across seeds —
# small IQR means stabble consistent results, large IQR means high variance.
def median_iqr(arr):
    q1 = np.percentile(arr, 25)  # lower quartile
    q3 = np.percentile(arr, 75)  # upper quartile
    return np.median(arr), q3 - q1  # median and spread