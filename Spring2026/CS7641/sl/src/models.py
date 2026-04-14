"""
Model training, tuning, and evaluation.

What does this code do:

- Trains and tunes Decision Trees, kNN, SVMs (linear + RBF), and Neural Networks (sklearn + PyTorch)
- Handles hyperparameter tuning via cross-validation
- Uses optimized implementations for speed (SGD for large datasets, caching for RBF)
- Returns trained models ready for evaluation

Key optimizations:
- Linear SVM auto-selects between SGD (large data) and LinearSVC (medium data)
- RBF SVM uses reduced grids, caching, and relaxed tolerence to run in ~5-10 mins on laptop
- PyTorch models use early stopping to avoid unnecesary epochs
"""

##########################

"""
Assignment Requirements Covered:

- All 4 required algorithms: Decision Trees, kNN, SVM (linear + RBF), Neural Networks
- Pruning/regularization for DT: max_depth, min_samples_leaf, ccp_alpha
- kNN:  multiple k values (3,5,11,21) with distance weighting
 - SVM: ≥2 kernels per dataset (linear + RBF), tuning C and gamma
- Neural Networks: both sklearn MLPClassifier and custom PyTorch implementation
- SGD only for NNs (no momentum, no adaptive optimizers like Adam/RMSprop)
-  Capacity scaling: shallow-wide vs deep-narrow architectures
- Early stopping and regularization for neural networks
-  Cross-validation for hyperparameter tuning on training data only
- Optimization strategies: grid reduction,  kernel caching, tolerence adjustment
"""

import numpy as np  # array operations
import warnings  # supress annoying convergence warnings
import time  # track how long RBF SVM takes
from sklearn.tree import DecisionTreeClassifier  # decision trees
from sklearn.neighbors import KNeighborsClassifier  # k nearest neighbors
from sklearn.svm import SVC, LinearSVC  # support vector machines
from sklearn.linear_model import SGDClassifier  # fast linear SVM via stochastic gradient descent
from sklearn.neural_network import MLPClassifier  # sklearn's neural net
from sklearn.model_selection import StratifiedKFold  # for cross-validation splits
from sklearn.exceptions import ConvergenceWarning  # to supress warnings
import torch  # PyTorch for custom neural nets
import torch.nn as nn  # neural network modules
import torch.optim as optim  # optimizers like SGD
from torch.utils.data import DataLoader, TensorDataset  # data loading utilities

from .utils import evaluate_model  # our custom evaluation function

# Supress convergence warnings globaly - they're noisy and don't help much
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)



# Helper function for cross-validation scoring
# Computes mean CV score across folds for a given estimator
def _cv_score(estimator, X_train, y_train, cv, scoring):
    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        # Clone the estimator so we start fresh each fold
        est = estimator.__class__(**estimator.get_params())
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X_train[train_idx], y_train[train_idx])
            y_pred = est.predict(X_train[val_idx])
            
        # Score based on task type
        if scoring == 'f1':
            s = evaluate_model(y_train[val_idx], y_pred, task='binary')[1]  # binary F1
        elif scoring == 'f1_macro':
            s = evaluate_model(y_train[val_idx], y_pred, task='multiclass')[1]  # macro F1
        else:
            s = evaluate_model(y_train[val_idx], y_pred, task='binary')[1]
        scores.append(s)
    return np.mean(scores)



# Tune max_depth and ccp_alpha for Decision Tree
# Returns model trained on full data with best params
def tune_decision_tree(X_train, y_train, cv, scoring):
    depth_list = [3, 5, 10, 15, 20, None]  # shallow to deep, plus unlimited
    alpha_list = np.logspace(-4, -1, 4)  # post-pruning strength
    best_params = None
    best_score = -1

    for depth in depth_list:
        for alpha in alpha_list:
            dt = DecisionTreeClassifier(max_depth=depth, ccp_alpha=alpha, random_state=42)
            score = _cv_score(dt, X_train, y_train, cv, scoring)
            if score > best_score:
                best_score = score
                best_params = {'max_depth': depth, 'ccp_alpha': alpha}

    # Retrain on full data with best params
    final_model = DecisionTreeClassifier(**best_params, random_state=42)
    final_model.fit(X_train, y_train)
    return final_model



# Tune n_neighbors for kNN
# Distance weighting helps with class imbalance
def tune_knn(X_train, y_train, cv, scoring):
    k_range = [3, 5, 11, 21]  # small k = low bias, large k = low variance
    best_k = None
    best_score = -1

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1)  # use all cores
        score = _cv_score(knn, X_train, y_train, cv, scoring)
        if score > best_score:
            best_score = score
            best_k = k

    final_model = KNeighborsClassifier(n_neighbors=best_k, weights='distance', n_jobs=-1)
    final_model.fit(X_train, y_train)
    return final_model


# Ultra-fast linear SVM using SGD - O(n) complexity
# Best for large datasets like Adult (30k+ samples)
def tune_svm_sgd(X_train, y_train, cv, scoring):
    unique_classes = len(np.unique(y_train))
    
    alpha_list = [0.0001, 0.001]  # regularization strength (smaller = less regularization)
    best_params = None
    best_score = -1
    best_model = None
    
    for alpha in alpha_list:
        sgd = SGDClassifier(
            loss='hinge',  # SVM loss
            penalty='l2',  # L2 regularization
            alpha=alpha,
            learning_rate='optimal',  # adaptively computed
            max_iter=1000,
            tol=1e-3,
            n_jobs=-1,
            early_stopping=True,  # stops if validation score doesn't improve
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=42
        )
        
        score = _cv_score(sgd, X_train, y_train, cv, scoring)
        
        if score > best_score:
            best_score = score
            best_params = {'alpha': alpha}
            # Retrain with best alpha
            best_model = SGDClassifier(
                loss='hinge',
                penalty='l2',
                alpha=alpha,
                learning_rate='optimal',
                max_iter=1000,
                tol=1e-3,
                n_jobs=-1,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=5,
                random_state=42
            )
            best_model.fit(X_train, y_train)
    
    return best_model



# Fast LinearSVC - O(n_features * n_samples)
# Best for medium-sized datasets like Wine
def tune_svm_linear_fast(X_train, y_train, cv, scoring):
    unique_classes = len(np.unique(y_train))
    C_range = np.logspace(-2, 2, 3)  # inverse regularization (larger C = less regularization)
    best_params = None
    best_score = -1
    best_model = None

    for C in C_range:
        svm = LinearSVC(
            C=C,
            loss='squared_hinge',  # smooth loss function
            dual=False,  # primal formulation for high-dimensional data
            random_state=42,
            max_iter=2000,
            tol=1e-3,
            multi_class='ovr'  # one-vs-rest for multiclass
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = _cv_score(svm, X_train, y_train, cv, scoring)
            
        if score > best_score:
            best_score = score
            best_params = {'C': C}
            best_model = LinearSVC(
                C=C,
                loss='squared_hinge',
                dual=False,
                random_state=42,
                max_iter=2000,
                tol=1e-3,
                multi_class='ovr'
            )
            best_model.fit(X_train, y_train)

    return best_model

# Auto-selects fastest linear SVM based on dataset size
# Large datasets get SGD, medium datasets get LinearSVC
def tune_svm_linear(X_train, y_train, cv, scoring):
    n_samples = X_train.shape[0]
    
    print(f"  Dataset size: {n_samples} samples")
    
    if n_samples > 10000:
        print("  Using SGD SVM (fastest for large datasets)")
        return tune_svm_sgd(X_train, y_train, cv, scoring)
    else:
        print("  Using LinearSVC with squared_hinge")
        return tune_svm_linear_fast(X_train, y_train, cv, scoring)




# Optimized RBF SVM for laptop hardware
# Runs in 5-10 minutes on Adult, 20-30 seconds on Wine
# Key optimizations: reduced grid, caching, relaxed tolerence
def tune_svm_rbf(X_train, y_train, cv, scoring):
    start_time = time.time()
    n_samples = X_train.shape[0]
    
    print(f"  Dataset size: {n_samples} samples")
    
    # Adult dataset (large) - agressive optimizations
    if n_samples > 10000:
        print("  Adult dataset: Using optimized RBF SVM (target: 5-8 minutes)")
        
        # Reduced parameter grid (3×2 = 6 combinations instead of 25+)
        C_range = [0.1, 1.0, 10.0]
        gamma_range = ['scale', 'auto']  # 'scale' = 1/(n_features*X.var()), 'auto' = 1/n_features
        
        # Performance optimizations
        cache_size = 1024  # 1GB kernel cache to avoid recomputation
        tolerance = 1e-3  # relaxed from 1e-4, gives ~10x speedup with minimal accuracy loss
        max_iterations = 3000  # for tuning phase
        shrinking = True  # heuristic to speed up training
        
        print(f"  Testing {len(C_range)}×{len(gamma_range)} = {len(C_range)*len(gamma_range)} combinations")
        print(f"  Cache: {cache_size}MB, Tolerance: {tolerance}, Max iterations: {max_iterations}")
        
    # Wine dataset (medium) - less agressive optimizations needed
    else:
        print("  Wine dataset: Using optimized RBF SVM (target: 20-30 seconds)")
        
        C_range = np.logspace(-2, 2, 3)
        gamma_range = ['scale', 'auto', 0.1]
        
        cache_size = 512
        tolerance = 1e-3
        max_iterations = 5000
        shrinking = True
        
        print(f"  Testing {len(C_range)}×{len(gamma_range)} = {len(C_range)*len(gamma_range)} combinations")
    
    best_score = -1
    best_params = None
    
    # Progress tracking
    total_combinations = len(C_range) * len(gamma_range)
    current = 0
    fold_times = []
    
    for C in C_range:
        for gamma in gamma_range:
            current += 1
            print(f"    Progress: {current}/{total_combinations} (C={C:.4f}, gamma={gamma})", end=' ' * 20 + '\r')
            
            svm = SVC(
                kernel='rbf',
                C=C,
                gamma=gamma,
                probability=False,  # speeds up training
                random_state=42,
                cache_size=cache_size,
                max_iter=max_iterations,
                tol=tolerance,
                shrinking=shrinking,
                decision_function_shape='ovr',
                verbose=False
            )
            
            fold_start = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = _cv_score(svm, X_train, y_train, cv, scoring)
            fold_time = time.time() - fold_start
            fold_times.append(fold_time)
            
            # Estimate remaining time after first combo
            if current == 1 and len(fold_times) > 0:
                avg_time_per_combo = np.mean(fold_times)
                remaining = avg_time_per_combo * (total_combinations - current)
                print(f"    Estimated remaining: {remaining:.1f} seconds", end='')
            
            if score > best_score:
                best_score = score
                best_params = {'C': C, 'gamma': gamma}
    
    print()
    
    # Display timing stats
    elapsed = time.time() - start_time
    print(f"  Tuning completed in {elapsed:.1f} seconds")
    print(f"  Best params: C={best_params['C']}, gamma={best_params['gamma']}")
    print(f"  Best CV score: {best_score:.4f}")
    
    # Train final model with best params on full dataset
    print(f"  Training final model on full dataset...")
    train_start = time.time()
    
    # Use more iterations for final model since we're not tuning anymore
    if n_samples > 10000:
        final_max_iter = 15000
    else:
        final_max_iter = 10000
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        best_model = SVC(
            kernel='rbf',
            **best_params,
            probability=(scoring == 'f1'),  # enable probability for calibration later
            random_state=42,
            cache_size=cache_size,
            max_iter=final_max_iter,
            tol=tolerance,
            shrinking=shrinking,
            decision_function_shape='ovr',
            verbose=False
        )
        best_model.fit(X_train, y_train)
    
    train_time = time.time() - train_start
    total_time = time.time() - start_time
    
    print(f"  Final training: {train_time:.1f} seconds")
    print(f"  TOTAL RBF SVM TIME: {total_time:.1f} seconds")
    print(f"  RBF SVM completed succesfully!")
    
    return best_model


# Tune sklearn's MLPClassifier
# Tests different hidden layer sizes and L2 regularization (alpha)
def tune_sklearn_nn(X_train, y_train, cv, scoring):
    sizes_list = [(50,), (100,)]  # shallow-wide architectures
    alpha_list = [0.0001, 0.001]  # L2 penalty strength
    best_params = None
    best_score = -1
    best_model = None

    for sizes in sizes_list:
        for alpha in alpha_list:
            mlp = MLPClassifier(
                hidden_layer_sizes=sizes,
                alpha=alpha,
                learning_rate_init=0.01,  # initial learning rate
                max_iter=200,
                random_state=42,
                solver='sgd',  # stochastic gradient descent
                momentum=0,  # no momentum for fair comparison with PyTorch
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=5  # stop if no improvement for 5 iterations
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = _cv_score(mlp, X_train, y_train, cv, scoring)
                
            if score > best_score:
                best_score = score
                best_params = {'hidden_layer_sizes': sizes, 'alpha': alpha}
                best_model = MLPClassifier(
                    hidden_layer_sizes=sizes,
                    alpha=alpha,
                    learning_rate_init=0.01,
                    max_iter=200,
                    random_state=42,
                    solver='sgd',
                    momentum=0,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=5
                )
                best_model.fit(X_train, y_train)

    return best_model


# Simple MLP architecture in PyTorch
# Takes input_dim -> hidden layers with ReLU -> output_dim
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))  # fully connected layer
            layers.append(nn.ReLU())  # activation function
            prev = h
        layers.append(nn.Linear(prev, output_dim))  # final output layer
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Train a PyTorch MLP with SGD optimizer
# Includes early stopping based on validation loss
def train_pytorch_model(X_train, y_train, X_val, y_val,
                        input_dim, hidden_dims, output_dim,
                        lr=0.01, weight_decay=0.0, epochs=100,
                        batch_size=64, early_stopping=5, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use GPU if availble
    model = SimpleMLP(input_dim, hidden_dims, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()  # standard classification loss
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0)

    # Convert numpy arrays to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()  # set to training mode
        train_loss = 0.0
        correct = 0
        total = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()  # clear gradients
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()  # backprop
            optimizer.step()  # update weights
            train_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        train_loss /= len(train_loader.dataset)
        train_acc = correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validation phase
        model.eval()  # set to evaluation mode
        with torch.no_grad():  # no gradient computation
            val_outputs = model(X_val_t.to(device))
            val_loss = criterion(val_outputs, y_val_t.to(device)).item()
            _, val_pred = torch.max(val_outputs, 1)
            val_acc = (val_pred.cpu() == y_val_t).sum().item() / len(y_val_t)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}  # save best weights
        elif epoch - best_epoch >= early_stopping:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break
    
    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)
    model = model.to(device)
    return model, history


# Tune PyTorch NN over learning rate, weight decay, and architecture
# Returns best model, training history, and config
def tune_pytorch_nn(X_train, y_train, X_val, y_val, input_dim, output_dim,
                    lr_list=[0.01, 0.001], wd_list=[0, 1e-4], hidden_archs=[[100], [50, 50]]):
    best_model = None
    best_score = -1
    best_history = None
    best_config = {}
    
    for hidden in hidden_archs:
        for lr in lr_list:
            for wd in wd_list:
                model, history = train_pytorch_model(
                    X_train, y_train, X_val, y_val,
                    input_dim, hidden, output_dim,
                    lr=lr, weight_decay=wd, epochs=100, batch_size=64, early_stopping=5)
                final_val_acc = history['val_acc'][-1]  # use final validation accuracy
                if final_val_acc > best_score:
                    best_score = final_val_acc
                    best_model = model
                    best_history = history
                    best_config = {'hidden': hidden, 'lr': lr, 'wd': wd}
    return best_model, best_history, best_config