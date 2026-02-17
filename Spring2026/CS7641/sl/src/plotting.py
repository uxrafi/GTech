"""
Plotting functions for learning curves, validation curves, confusion matricies.

What it does:
- Generates learning curves to diagnose bias/variance tradeoffs
- Creates validation curves to visualize hyperparameter effects
- Plots confusion matricies to see  classification errors
- Auto-saves figures in outputs/figures

All plots use red for training, green for validation and include standard deviations as shaded regions for uncertanty visualization.
"""

##########################

"""
Assignment Requirments Covered:

- Learning curves (LC): training and validation metric vs training size
- Model-complexity curves (MC): validation metric vs hyperparameter
- Confusion matricies for classification tasks
- Supports "Per-algorithm required figures/tables" requirement
- Bias/variance diagnostics via learning curve gaps
- Shaded regions show uncertainty (std dev) across CV folds
"""

import numpy as np  # array math for means and std devs
import matplotlib.pyplot as plt  # plotting libary
import seaborn as sns  # makes nicer heatmaps for confusion matricies
from sklearn.model_selection import learning_curve, validation_curve  # sklearn's curve generators
from sklearn.metrics import confusion_matrix  # compute confusion matrix




# Plot learning curve showing how performance changes with training set size
# Helps diagnose if model is overfiting (high variance) or underfiting (high bias)
def plot_learning_curve(estimator, X, y, title, scoring, cv=3, train_sizes=np.linspace(0.1,1.0,5), save_path=None):
    # Generate learning curve data using cross-validation
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring,
        train_sizes=train_sizes, n_jobs=-1, random_state=42)  # use all CPU cores
    
    # Compute mean and std dev across CV folds
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure()
    plt.title(f"Learning Curve ({title})")
    plt.xlabel("Training examples")
    plt.ylabel(scoring)
    plt.grid()  # add grid for readabilty
    
    # Shaded regions show variance across folds
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    
    # Plot mean scores
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')  # high-res save
    plt.close()  # free memory
    return



# Plot validation curve showing how performance changes with a single hyperparamter
# Helps identify optimal parameter value and diagnose overfiting/underfiting
def plot_validation_curve(estimator, X, y, param_name, param_range, title, scoring, cv=3, save_path=None):
    # Generate validation curve data
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=-1)
    
    # Compute mean and std dev across folds
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure()
    plt.title(f"Validation Curve ({title})")
    plt.xlabel(param_name)
    plt.ylabel(scoring)
    plt.grid()
    
    # Shaded uncertanty bands
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    
    # Plot mean scores across parameter range
    plt.plot(param_range, train_mean, 'o-', color="r", label="Training score")
    plt.plot(param_range, test_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return



# Plot confusion matrix as a heatmap
# Shows where model makes mistakes - diagonal is correct predictions
def plot_confusion_matrix(y_true, y_pred, labels, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred)  # compute matrix
    
    plt.figure()
    # Heatmap with annotations showing counts
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title(title)
    plt.ylabel('True')  # actual labels on y-axis
    plt.xlabel('Predicted')  # predicted labels on x-axis
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return