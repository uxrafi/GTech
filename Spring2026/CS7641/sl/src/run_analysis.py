#!/usr/bin/env python  # shebang for Linux execution - code developed on Windows but tested to be compatible


"""
Main script to run the complete supervised learning analisys.

What this does:
- Loads Adult and Wine datasets with preprocessing
- Trains and tunes all models (DT, kNN, SVM linear/RBF, NN sklearn/PyTorch)
- Generates learning curves, validation curves, and confusion matricies
- Saves all results to JSON files and figures to outputs/figures/

Execution order: Wine first (faster, 2-3 mins) then Adult (slower, 10-15 mins)
This ensures we get partial results even if Adult RBF SVM times out.
"""

##########################

"""
Assignment Requirements Covered:

- Complete workflow: EDA → Preprocessing → Train → Tune → Evaluate
- Single held-out test split (80/20) with stratification
- Cross-validation (StratifiedKFold, n=3) for tuning on training data only
- All required metrics:
  * Adult: F1, Accuracy, PR-AUC (for imbalanced binary classification)
  * Wine: Macro-F1, Accuracy (for imbalanced multiclass)
- Confusion matricies at justified operating points
- Runtime profiling: fit_time and pred_time via wall-clock measurment
- Learning curves for all models (diagnose bias/variance)
- Validation/complexity curves for all models (hyperparameter effects)
- Epoch-based curves for both NN implementations (training progress)
- Width complexity comparision for NNs (capacity scaling)
- Direct comparision between sklearn and PyTorch NN implementations
- All figures auto-saved to outputs/figures/
- Results saved to JSON for reproducability
- Fixed random seeds throughout for deterministic results
"""



import sys  # system-level operations (not actualy used but kept for compatability)
import json  # save results as JSON files
import numpy as np  # array math
from sklearn.model_selection import StratifiedKFold, train_test_split  # CV splits and train/test splitting
from sklearn.metrics import f1_score, precision_recall_curve, auc  # evaluation metrix
from sklearn.svm import LinearSVC  # linear support vector machine
from sklearn.linear_model import SGDClassifier  # stochastic gradient descent classifier
from sklearn.tree import DecisionTreeClassifier  # decision trees
from sklearn.neighbors import KNeighborsClassifier  # k nearest neighbors
from sklearn.svm import SVC  # support vector classifier (for RBF kernel)
from sklearn.neural_network import MLPClassifier  # sklearn's neural network
import torch  # PyTorch for custom neural nets
import matplotlib.pyplot as plt  # plotting libary

from src.paths import RANDOM_STATE, FIGURES_DIR, OUTPUT_DIR  # centralized paths and seed
from src.utils import set_seeds, evaluate_model, measure_time  # utility functions
from src.data_loader import load_adult, load_wine, get_adult_preprocessor, get_wine_preprocessor, split_data  # data loading
from src import models  # our model training functions
from src.plotting import plot_learning_curve, plot_validation_curve, plot_confusion_matrix  # plotting utilities

set_seeds(RANDOM_STATE)  # ensure reproducability across all libraries


# Run complete analysis on Adult dataset
# Binary classification: income <=50K vs >50K
def run_adult():
    print("=== Adult Dataset ===")
    X, y = load_adult()
    preprocessor = get_adult_preprocessor()
    X_train, X_test, y_train, y_test = split_data(X, y, stratify=True)

    # Fit preprocessor only on training data to avoid leakage
    preprocessor.fit(X_train)
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scoring = 'f1'  # primary metric for imbalanced binary classification
    results = {}

    # Decision Tree
    print("Training Decision Tree...")
    dt = models.tune_decision_tree(X_train_proc, y_train.values, cv, scoring)
    fit_t, pred_t, y_pred = measure_time(dt, X_train_proc, y_train, X_test_proc)
    y_prob = dt.predict_proba(X_test_proc)
    acc, f1, pr_auc = evaluate_model(y_test, y_pred, y_prob, task='binary')
    results['DT'] = {'f1': f1, 'pr_auc': pr_auc, 'acc': acc,
                     'fit_time': fit_t, 'pred_time': pred_t,
                     'y_pred': y_pred.tolist(), 'y_prob': y_prob.tolist(),
                     'best_params': {'max_depth': dt.max_depth, 'ccp_alpha': dt.ccp_alpha}}
    
    # Generate plots
    plot_learning_curve(dt, X_train_proc, y_train, 'DT (Adult)', scoring='f1', cv=cv,
                        save_path=FIGURES_DIR/'adult_dt_learning.png')
    plot_validation_curve(DecisionTreeClassifier(ccp_alpha=1e-4, random_state=RANDOM_STATE),
                          X_train_proc, y_train, 'max_depth', [3,5,10,15,20],
                          'DT (Adult) - Max Depth', scoring='f1', cv=cv,
                          save_path=FIGURES_DIR/'adult_dt_complexity.png')
    plot_confusion_matrix(y_test, y_pred, ['<=50K','>50K'], 'DT (Adult)',
                          save_path=FIGURES_DIR/'adult_cm_dt.png')
    print("  DT done.")

    # kNN
    print("Training kNN...")
    knn = models.tune_knn(X_train_proc, y_train.values, cv, scoring)
    fit_t, pred_t, y_pred = measure_time(knn, X_train_proc, y_train, X_test_proc)
    y_prob = knn.predict_proba(X_test_proc)
    acc, f1, pr_auc = evaluate_model(y_test, y_pred, y_prob, task='binary')
    results['kNN'] = {'f1': f1, 'pr_auc': pr_auc, 'acc': acc,
                      'fit_time': fit_t, 'pred_time': pred_t,
                      'y_pred': y_pred.tolist(), 'y_prob': y_prob.tolist(),
                      'best_params': {'k': knn.n_neighbors}}
    plot_learning_curve(knn, X_train_proc, y_train, 'kNN (Adult)', scoring='f1', cv=cv,
                        save_path=FIGURES_DIR/'adult_knn_learning.png')
    plot_validation_curve(KNeighborsClassifier(weights='distance'),
                          X_train_proc, y_train, 'n_neighbors', [3,5,11,21],
                          'kNN (Adult)', scoring='f1', cv=cv,
                          save_path=FIGURES_DIR/'adult_knn_complexity.png')
    plot_confusion_matrix(y_test, y_pred, ['<=50K','>50K'], 'kNN (Adult)',
                          save_path=FIGURES_DIR/'adult_cm_knn.png')
    print("  kNN done.")

    # Linear SVM (auto-selects between SGD and LinearSVC based on data size)
    print("Training Linear SVM...")
    svm_lin = models.tune_svm_linear(X_train_proc, y_train.values, cv, scoring)
    fit_t, pred_t, y_pred = measure_time(svm_lin, X_train_proc, y_train, X_test_proc)
    
    # Handle different SVM types for prediction scores
    if hasattr(svm_lin, 'decision_function'):
        y_scores = svm_lin.decision_function(X_test_proc)
        # For binary classification, ensure correct shape
        if len(y_scores.shape) == 1:
            y_scores = y_scores.reshape(-1, 1)
    else:
        y_scores = None
    
    acc, f1 = evaluate_model(y_test, y_pred, task='binary')
    
    # Calculate PR-AUC if we have decision scores
    if y_scores is not None and y_scores.shape[1] > 1:
        precision, recall, _ = precision_recall_curve(y_test, y_scores[:, 1])
        pr_auc = auc(recall, precision)
    elif y_scores is not None:
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        pr_auc = auc(recall, precision)
    else:
        pr_auc = 0.0
    
    # Get best params (different model types have different param names)
    if hasattr(svm_lin, 'C'):
        best_param_name = 'C'
        best_param_value = svm_lin.C
    elif hasattr(svm_lin, 'alpha'):
        best_param_name = 'alpha'
        best_param_value = svm_lin.alpha
    else:
        best_param_name = 'C'
        best_param_value = 'N/A'
        
    results['SVM_Linear'] = {'f1': f1, 'pr_auc': pr_auc, 'acc': acc,
                              'fit_time': fit_t, 'pred_time': pred_t,
                              'y_pred': y_pred.tolist(),
                              'best_params': {best_param_name: best_param_value}}
    
    # Plots
    plot_learning_curve(svm_lin, X_train_proc, y_train, 'Linear SVM (Adult)', scoring='f1', cv=cv,
                        save_path=FIGURES_DIR/'adult_svm_linear_learning.png')
    
    # Validation curve using apropriate estimator type
    if isinstance(svm_lin, SGDClassifier):
        plot_validation_curve(SGDClassifier(loss='hinge', random_state=RANDOM_STATE),
                              X_train_proc, y_train, 'alpha', np.logspace(-4, -1, 4),
                              'Linear SVM (Adult)', scoring='f1', cv=cv,
                              save_path=FIGURES_DIR/'adult_svm_linear_complexity.png')
    else:
        plot_validation_curve(LinearSVC(loss='squared_hinge', dual=False, random_state=RANDOM_STATE),
                              X_train_proc, y_train, 'C', np.logspace(-2, 2, 5),
                              'Linear SVM (Adult)', scoring='f1', cv=cv,
                              save_path=FIGURES_DIR/'adult_svm_linear_complexity.png')
    
    plot_confusion_matrix(y_test, y_pred, ['<=50K','>50K'], 'Linear SVM (Adult)',
                          save_path=FIGURES_DIR/'adult_cm_svm_lin.png')
    print("  Linear SVM done.")

    # RBF SVM (this is the slow one - optimized to run in ~10 mins on laptop)
    print("Training RBF SVM...")
    svm_rbf = models.tune_svm_rbf(X_train_proc, y_train.values, cv, scoring)
    fit_t, pred_t, y_pred = measure_time(svm_rbf, X_train_proc, y_train, X_test_proc)
    y_prob = svm_rbf.predict_proba(X_test_proc)
    acc, f1, pr_auc = evaluate_model(y_test, y_pred, y_prob, task='binary')
    results['SVM_RBF'] = {'f1': f1, 'pr_auc': pr_auc, 'acc': acc,
                          'fit_time': fit_t, 'pred_time': pred_t,
                          'y_pred': y_pred.tolist(), 'y_prob': y_prob.tolist(),
                          'best_params': {'C': svm_rbf.C, 'gamma': svm_rbf.gamma}}
    plot_learning_curve(svm_rbf, X_train_proc, y_train, 'RBF SVM (Adult)', scoring='f1', cv=cv,
                        save_path=FIGURES_DIR/'adult_svm_rbf_learning.png')
    plot_validation_curve(SVC(kernel='rbf', gamma='scale', probability=True, random_state=RANDOM_STATE),
                          X_train_proc, y_train, 'C', np.logspace(-2,2,5),
                          'RBF SVM (Adult)', scoring='f1', cv=cv,
                          save_path=FIGURES_DIR/'adult_svm_rbf_complexity.png')
    plot_confusion_matrix(y_test, y_pred, ['<=50K','>50K'], 'RBF SVM (Adult)',
                          save_path=FIGURES_DIR/'adult_cm_svm_rbf.png')
    print("  RBF SVM done.")
    
    # Force memory cleanup after RBF SVM to avoid memory issues
    import gc
    gc.collect()
    del svm_rbf
    gc.collect()

    # sklearn Neural Network
    print("Training sklearn NN...")
    mlp = models.tune_sklearn_nn(X_train_proc, y_train.values, cv, scoring)
    fit_t, pred_t, y_pred = measure_time(mlp, X_train_proc, y_train, X_test_proc)
    y_prob = mlp.predict_proba(X_test_proc)
    acc, f1, pr_auc = evaluate_model(y_test, y_pred, y_prob, task='binary')
    results['NN_sklearn'] = {'f1': f1, 'pr_auc': pr_auc, 'acc': acc,
                              'fit_time': fit_t, 'pred_time': pred_t,
                              'y_pred': y_pred.tolist(), 'y_prob': y_prob.tolist(),
                              'best_params': {'hidden_layer_sizes': mlp.hidden_layer_sizes,
                                              'alpha': mlp.alpha}}
    plot_learning_curve(mlp, X_train_proc, y_train, 'sklearn NN (Adult)', scoring='f1', cv=cv,
                        save_path=FIGURES_DIR/'adult_nn_sklearn_learning.png')
    
    # Complexity curve for network width
    width_range = [10,50,100,200]
    train_scores, val_scores = [], []
    for w in width_range:
        mlp_temp = MLPClassifier(hidden_layer_sizes=(w,), alpha=0.001, learning_rate_init=0.01,
                                 max_iter=200, random_state=RANDOM_STATE, solver='sgd', momentum=0)
        scores = []
        for train_idx, val_idx in cv.split(X_train_proc, y_train):
            mlp_temp.fit(X_train_proc[train_idx], y_train.iloc[train_idx])
            y_pred_train = mlp_temp.predict(X_train_proc[train_idx])
            y_pred_val = mlp_temp.predict(X_train_proc[val_idx])
            train_scores_fold = f1_score(y_train.iloc[train_idx], y_pred_train)
            val_scores_fold = f1_score(y_train.iloc[val_idx], y_pred_val)
            scores.append((train_scores_fold, val_scores_fold))
        train_scores.append(np.mean([s[0] for s in scores]))
        val_scores.append(np.mean([s[1] for s in scores]))
    
    plt.figure()
    plt.plot(width_range, train_scores, 'o-', label='Train')
    plt.plot(width_range, val_scores, 'o-', label='Val')
    plt.xlabel('Width')
    plt.ylabel('F1')
    plt.title('sklearn NN (Adult) - Width Complexity')
    plt.legend()
    plt.grid()
    plt.savefig(FIGURES_DIR/'adult_nn_sklearn_complexity.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plot_confusion_matrix(y_test, y_pred, ['<=50K','>50K'], 'sklearn NN (Adult)',
                          save_path=FIGURES_DIR/'adult_cm_nn_sklearn.png')
    print("  sklearn NN done.")

    # PyTorch Neural Network
    print("Training PyTorch NN...")
    # Create validation split from training data for hyperparameter tuning
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_proc, y_train, test_size=0.2,
                                                  stratify=y_train, random_state=RANDOM_STATE)
    input_dim = X_tr.shape[1]
    output_dim = 2  # Binary classification for Adult
    pt_model, pt_history, pt_config = models.tune_pytorch_nn(
        X_tr, y_tr.values, X_val, y_val.values,
        input_dim, output_dim,
        lr_list=[0.01,0.001], wd_list=[0,1e-4],
        hidden_archs=[[100],[50,50]])
    
    # Measure performance on test set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_test_t = torch.tensor(X_test_proc, dtype=torch.float32).to(device)
    pt_model.eval()
    import time
    start = time.time()
    with torch.no_grad():  # no gradient computation during inference
        outputs = pt_model(X_test_t)
        _, y_pred_pt = torch.max(outputs, 1)
    pred_time_pt = time.time() - start
    y_pred_pt = y_pred_pt.cpu().numpy()
    y_prob_pt = torch.softmax(outputs, dim=1).cpu().numpy()
    acc_pt, f1_pt, pr_auc_pt = evaluate_model(y_test, y_pred_pt, y_prob_pt, task='binary')
    results['NN_pytorch'] = {'f1': f1_pt, 'pr_auc': pr_auc_pt, 'acc': acc_pt,
                              'fit_time': None, 'pred_time': pred_time_pt,
                              'y_pred': y_pred_pt.tolist(), 'y_prob': y_prob_pt.tolist(),
                              'best_params': pt_config}
    
    # Plot training history (loss and accuracy over epochs)
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    axes[0].plot(pt_history['train_loss'], label='Train')
    axes[0].plot(pt_history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('PyTorch NN (Adult) - Loss'); axes[0].legend()
    axes[1].plot(pt_history['train_acc'], label='Train')
    axes[1].plot(pt_history['val_acc'], label='Val')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('PyTorch NN (Adult) - Accuracy'); axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/'adult_nn_pytorch_epochs.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Complexity curve for PyTorch network width
    widths = [10,50,100,200]
    val_accs = []
    for w in widths:
        model, hist = models.train_pytorch_model(
            X_tr, y_tr.values, X_val, y_val.values,
            input_dim, [w], output_dim,
            lr=0.01, weight_decay=0, epochs=50, batch_size=64, early_stopping=3)
        val_accs.append(hist['val_acc'][-1])
    
    plt.figure()
    plt.plot(widths, val_accs, 'o-')
    plt.xlabel('Width')
    plt.ylabel('Validation Accuracy')
    plt.title('PyTorch NN (Adult) - Width Complexity')
    plt.grid()
    plt.savefig(FIGURES_DIR/'adult_nn_pytorch_complexity.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plot_confusion_matrix(y_test, y_pred_pt, ['<=50K','>50K'], 'PyTorch NN (Adult)',
                          save_path=FIGURES_DIR/'adult_cm_nn_pytorch.png')
    print("  PyTorch NN done.")

    # Save all results to JSON
    with open(OUTPUT_DIR/'adult_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Adult results saved.\n")


# Run complete analisys on Wine dataset
# Multiclass classification: quality ratings 3-9
def run_wine():
    print("=== Wine Dataset ===")
    X, y = load_wine()
    preprocessor = get_wine_preprocessor()
    X_train, X_test, y_train, y_test = split_data(X, y, stratify=True)

    # Preprocess
    preprocessor.fit(X_train)
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scoring = 'f1_macro'  # macro-averaged F1 for multiclass imbalanced data
    results = {}

    # Decision Tree
    print("Training Decision Tree...")
    dt = models.tune_decision_tree(X_train_proc, y_train, cv, scoring)
    fit_t, pred_t, y_pred = measure_time(dt, X_train_proc, y_train, X_test_proc)
    acc, f1_macro = evaluate_model(y_test, y_pred, task='multiclass')
    results['DT'] = {'f1_macro': f1_macro, 'acc': acc,
                     'fit_time': fit_t, 'pred_time': pred_t,
                     'y_pred': y_pred.tolist(),
                     'best_params': {'max_depth': dt.max_depth, 'ccp_alpha': dt.ccp_alpha}}
    plot_learning_curve(dt, X_train_proc, y_train, 'DT (Wine)', scoring='f1_macro', cv=cv,
                        save_path=FIGURES_DIR/'wine_dt_learning.png')
    plot_validation_curve(DecisionTreeClassifier(ccp_alpha=1e-4, random_state=RANDOM_STATE),
                          X_train_proc, y_train, 'max_depth', [3,5,10,15,20],
                          'DT (Wine) - Max Depth', scoring='f1_macro', cv=cv,
                          save_path=FIGURES_DIR/'wine_dt_complexity.png')
    plot_confusion_matrix(y_test, y_pred, sorted(np.unique(y_test)), 'DT (Wine)',
                          save_path=FIGURES_DIR/'wine_cm_dt.png')
    print("  DT done.")

    # kNN
    print("Training kNN...")
    knn = models.tune_knn(X_train_proc, y_train, cv, scoring)
    fit_t, pred_t, y_pred = measure_time(knn, X_train_proc, y_train, X_test_proc)
    acc, f1_macro = evaluate_model(y_test, y_pred, task='multiclass')
    results['kNN'] = {'f1_macro': f1_macro, 'acc': acc,
                      'fit_time': fit_t, 'pred_time': pred_t,
                      'y_pred': y_pred.tolist(),
                      'best_params': {'k': knn.n_neighbors}}
    plot_learning_curve(knn, X_train_proc, y_train, 'kNN (Wine)', scoring='f1_macro', cv=cv,
                        save_path=FIGURES_DIR/'wine_knn_learning.png')
    plot_validation_curve(KNeighborsClassifier(weights='distance'),
                          X_train_proc, y_train, 'n_neighbors', [3,5,11,21],
                          'kNN (Wine)', scoring='f1_macro', cv=cv,
                          save_path=FIGURES_DIR/'wine_knn_complexity.png')
    plot_confusion_matrix(y_test, y_pred, sorted(np.unique(y_test)), 'kNN (Wine)',
                          save_path=FIGURES_DIR/'wine_cm_knn.png')
    print("  kNN done.")

    # Linear SVM
    print("Training Linear SVM...")
    svm_lin = models.tune_svm_linear(X_train_proc, y_train, cv, scoring)
    fit_t, pred_t, y_pred = measure_time(svm_lin, X_train_proc, y_train, X_test_proc)
    acc, f1_macro = evaluate_model(y_test, y_pred, task='multiclass')
    
    # Get best params (handel different model types)
    if hasattr(svm_lin, 'C'):
        best_param_name = 'C'
        best_param_value = svm_lin.C
    elif hasattr(svm_lin, 'alpha'):
        best_param_name = 'alpha'
        best_param_value = svm_lin.alpha
    else:
        best_param_name = 'C'
        best_param_value = 'N/A'
        
    results['SVM_Linear'] = {'f1_macro': f1_macro, 'acc': acc,
                              'fit_time': fit_t, 'pred_time': pred_t,
                              'y_pred': y_pred.tolist(),
                              'best_params': {best_param_name: best_param_value}}
    
    plot_learning_curve(svm_lin, X_train_proc, y_train, 'Linear SVM (Wine)', scoring='f1_macro', cv=cv,
                        save_path=FIGURES_DIR/'wine_svm_linear_learning.png')
    
    # Validation curve using apropriate estimator
    if isinstance(svm_lin, SGDClassifier):
        plot_validation_curve(SGDClassifier(loss='hinge', random_state=RANDOM_STATE),
                              X_train_proc, y_train, 'alpha', np.logspace(-4, -1, 4),
                              'Linear SVM (Wine)', scoring='f1_macro', cv=cv,
                              save_path=FIGURES_DIR/'wine_svm_linear_complexity.png')
    else:
        plot_validation_curve(LinearSVC(loss='squared_hinge', dual=False, random_state=RANDOM_STATE),
                              X_train_proc, y_train, 'C', np.logspace(-2, 2, 5),
                              'Linear SVM (Wine)', scoring='f1_macro', cv=cv,
                              save_path=FIGURES_DIR/'wine_svm_linear_complexity.png')
    
    plot_confusion_matrix(y_test, y_pred, sorted(np.unique(y_test)), 'Linear SVM (Wine)',
                          save_path=FIGURES_DIR/'wine_cm_svm_lin.png')
    print("  Linear SVM done.")

    # RBF SVM
    print("Training RBF SVM...")
    svm_rbf = models.tune_svm_rbf(X_train_proc, y_train, cv, scoring)
    fit_t, pred_t, y_pred = measure_time(svm_rbf, X_train_proc, y_train, X_test_proc)
    acc, f1_macro = evaluate_model(y_test, y_pred, task='multiclass')
    results['SVM_RBF'] = {'f1_macro': f1_macro, 'acc': acc,
                          'fit_time': fit_t, 'pred_time': pred_t,
                          'y_pred': y_pred.tolist(),
                          'best_params': {'C': svm_rbf.C, 'gamma': svm_rbf.gamma}}
    plot_learning_curve(svm_rbf, X_train_proc, y_train, 'RBF SVM (Wine)', scoring='f1_macro', cv=cv,
                        save_path=FIGURES_DIR/'wine_svm_rbf_learning.png')
    plot_validation_curve(SVC(kernel='rbf', gamma='scale', random_state=RANDOM_STATE),
                          X_train_proc, y_train, 'C', np.logspace(-2,2,5),
                          'RBF SVM (Wine)', scoring='f1_macro', cv=cv,
                          save_path=FIGURES_DIR/'wine_svm_rbf_complexity.png')
    plot_confusion_matrix(y_test, y_pred, sorted(np.unique(y_test)), 'RBF SVM (Wine)',
                          save_path=FIGURES_DIR/'wine_cm_svm_rbf.png')
    print("  RBF SVM done.")

    # sklearn NN
    print("Training sklearn NN...")
    mlp = models.tune_sklearn_nn(X_train_proc, y_train, cv, scoring)
    fit_t, pred_t, y_pred = measure_time(mlp, X_train_proc, y_train, X_test_proc)
    acc, f1_macro = evaluate_model(y_test, y_pred, task='multiclass')
    results['NN_sklearn'] = {'f1_macro': f1_macro, 'acc': acc,
                              'fit_time': fit_t, 'pred_time': pred_t,
                              'y_pred': y_pred.tolist(),
                              'best_params': {'hidden_layer_sizes': mlp.hidden_layer_sizes,
                                              'alpha': mlp.alpha}}
    plot_learning_curve(mlp, X_train_proc, y_train, 'sklearn NN (Wine)', scoring='f1_macro', cv=cv,
                        save_path=FIGURES_DIR/'wine_nn_sklearn_learning.png')
    
    # Width complexity
    width_range = [10,50,100,200]
    train_scores, val_scores = [], []
    for w in width_range:
        mlp_temp = MLPClassifier(hidden_layer_sizes=(w,), alpha=0.001, learning_rate_init=0.01,
                                 max_iter=200, random_state=RANDOM_STATE, solver='sgd', momentum=0)
        scores = []
        for train_idx, val_idx in cv.split(X_train_proc, y_train):
            mlp_temp.fit(X_train_proc[train_idx], y_train[train_idx])
            y_pred_train = mlp_temp.predict(X_train_proc[train_idx])
            y_pred_val = mlp_temp.predict(X_train_proc[val_idx])
            train_scores_fold = f1_score(y_train[train_idx], y_pred_train, average='macro')
            val_scores_fold = f1_score(y_train[val_idx], y_pred_val, average='macro')
            scores.append((train_scores_fold, val_scores_fold))
        train_scores.append(np.mean([s[0] for s in scores]))
        val_scores.append(np.mean([s[1] for s in scores]))
    
    plt.figure()
    plt.plot(width_range, train_scores, 'o-', label='Train')
    plt.plot(width_range, val_scores, 'o-', label='Val')
    plt.xlabel('Width')
    plt.ylabel('Macro-F1')
    plt.title('sklearn NN (Wine) - Width Complexity')
    plt.legend(); plt.grid()
    plt.savefig(FIGURES_DIR/'wine_nn_sklearn_complexity.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plot_confusion_matrix(y_test, y_pred, sorted(np.unique(y_test)), 'sklearn NN (Wine)',
                          save_path=FIGURES_DIR/'wine_cm_nn_sklearn.png')
    print("  sklearn NN done.")

    # PyTorch NN - FIXED for wine quality range 0-9
    print("Training PyTorch NN...")
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_proc, y_train, test_size=0.2,
                                                  stratify=y_train, random_state=RANDOM_STATE)
    input_dim = X_tr.shape[1]
    
    # FIX: Wine quality ranges from 0-9, use 10 output classes to be safe
    output_dim = 10
    print(f"  Number of classes: {output_dim}")
    
    pt_model, pt_history, pt_config = models.tune_pytorch_nn(
        X_tr, y_tr, X_val, y_val,
        input_dim, output_dim,
        lr_list=[0.01,0.001], wd_list=[0,1e-4],
        hidden_archs=[[100],[50,50]])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_test_t = torch.tensor(X_test_proc, dtype=torch.float32).to(device)
    pt_model.eval()
    import time
    start = time.time()
    with torch.no_grad():
        outputs = pt_model(X_test_t)
        _, y_pred_pt = torch.max(outputs, 1)
    pred_time_pt = time.time() - start
    y_pred_pt = y_pred_pt.cpu().numpy()
    acc_pt, f1_macro_pt = evaluate_model(y_test, y_pred_pt, task='multiclass')
    results['NN_pytorch'] = {'f1_macro': f1_macro_pt, 'acc': acc_pt,
                              'fit_time': None, 'pred_time': pred_time_pt,
                              'y_pred': y_pred_pt.tolist(),
                              'best_params': pt_config}
    
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    axes[0].plot(pt_history['train_loss'], label='Train')
    axes[0].plot(pt_history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('PyTorch NN (Wine) - Loss'); axes[0].legend()
    axes[1].plot(pt_history['train_acc'], label='Train')
    axes[1].plot(pt_history['val_acc'], label='Val')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('PyTorch NN (Wine) - Accuracy'); axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/'wine_nn_pytorch_epochs.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Width complexity
    widths = [10,50,100,200]
    val_accs = []
    for w in widths:
        model, hist = models.train_pytorch_model(
            X_tr, y_tr, X_val, y_val,
            input_dim, [w], output_dim,
            lr=0.01, weight_decay=0, epochs=50, batch_size=64, early_stopping=3)
        val_accs.append(hist['val_acc'][-1])
    
    plt.figure()
    plt.plot(widths, val_accs, 'o-')
    plt.xlabel('Width')
    plt.ylabel('Validation Accuracy')
    plt.title('PyTorch NN (Wine) - Width Complexity')
    plt.grid()
    plt.savefig(FIGURES_DIR/'wine_nn_pytorch_complexity.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plot_confusion_matrix(y_test, y_pred_pt, sorted(np.unique(y_test)), 'PyTorch NN (Wine)',
                          save_path=FIGURES_DIR/'wine_cm_nn_pytorch.png')
    print("  PyTorch NN done.")

    with open(OUTPUT_DIR/'wine_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Wine results saved.\n")

# Main execution
# Run Wine first (faster) to get partial results even if Adult times out
if __name__ == '__main__':
    print("=" * 60)
    print("RUNNING WINE DATASET FIRST (faster, 2-3 minutes)")
    print("=" * 60)
    run_wine()
    
    print("\n" + "=" * 60)
    print("RUNNING ADULT DATASET SECOND (slower, 10-15 minutes)")
    print("=" * 60)
    run_adult()
    
    print("\n" + "=" * 60)
    print("ALL DONE! Figures saved in outputs/figures/, results in outputs/.")
    print("=" * 60)