"""
Model definitions for both SL and OL experiments.

What does this code do:

- SL models (unchanged): DecisionTree, kNN, SVM (linear + RBF), sklearn MLP
- OL additions:
  - SimpleMLP now accepts dropout_rate (defualt 0.0, fully backwards compatable)
  - AdamNoBiasCorrection: custom optimzer omitting bias-correction terms
  - _get_linear_layers, get/set_last_layer_params: RO layer access helpers
  - freeze_all_but_last: freezes all but last N linear layers for RO

Key things to note:
- The AdamNoBiasCorrection class was a pain to implement, make sure not to break it
- dropout_rate=0.0 by defualt so nothing in SL breaks when importing this
- freeze_all_but_last is used in OL Part 1 warm-starting experiments
"""

##########################

"""
Assignment Requirements Covered:

- All 4 required algorythms: Decision Trees, kNN, SVM (linear + RBF), Neural Networks
- OL Part 1: Randomized optimization with layer freezing and warm-start
- OL Part 2: Ablation study comparing Adam vs Adam without bias corection
- OL Part 3: Dropout regularization study (dropout_rate parameter)
- SGD only for sklearn NNs (no momentum, no adaptave optimizers)
- Cross-validation for hyperparemeter tuning on training data only
- Early stopping for both sklearn and PyTorch models to avoid unecessary epochs
- Capacity scaling: shallow-wide vs deep-narrow architechtures tested in tuning
"""

import numpy as np  # array operations
import warnings  # supress annoying convergance warnings
import time  # track timing (used in rbf svm)
from sklearn.tree import DecisionTreeClassifier  # decision trees
from sklearn.neighbors import KNeighborsClassifier  # k nearest neighbors
from sklearn.svm import SVC, LinearSVC  # support vector machines
from sklearn.linear_model import SGDClassifier  # fast linear SVM via stochastic gradient decent
from sklearn.neural_network import MLPClassifier  # sklearns neural net
from sklearn.model_selection import StratifiedKFold  # for cross-validation splits
from sklearn.exceptions import ConvergenceWarning  # to supress warnings
import torch  # PyTorch for custom neural nets
import torch.nn as nn  # neural network modules
import torch.optim as optim  # optimizers like SGD, Adam
from torch.utils.data import DataLoader, TensorDataset  # data loading utilitys

from utils import evaluate_model  # our custom evaluation function
from paths import RANDOM_STATE  # centralized random seed so results are reproducable

# Supress convergence warnings globaly - they clutter the output and arnt helpful
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Custom optimizer: Adam without bias correction (OL Part 2)
# ---------------------------------------------------------------------------
# PyTorch's built-in Adam always applys bias correction with no flag to
# disable it. This custom class omits the corection so we can isolate
# its effect in the ablation study. Took a while to get right - dont touch
class AdamNoBiasCorrection(optim.Optimizer):
    """Adam update without the (1-beta^t) bias-corection terms.

    PyTorch's built-in Adam always applies bias correction with no flag to
    disable it. This implementaion omits the correction so we can isolate
    its effect in the Part 2 ablation study.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr           = group['lr']
            beta1, beta2 = group['betas']
            eps          = group['eps']
            wd           = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamNoBiasCorrection does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state['step']       = 0
                    state['exp_avg']    = torch.zeros_like(p)  # first moment (mean)
                    state['exp_avg_sq'] = torch.zeros_like(p)  # second moment (variance)

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Apply weight decay if set (adds l2 penalty to gradient)
                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                # Update running moment estimates - standard exponential moving avg
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # No bias correction here - raw moment estimates used directly
                # This is the whole point of the class, dont add bias correction back!
                denom = exp_avg_sq.sqrt().add_(eps)
                p.addcdiv_(exp_avg, denom, value=-lr)

        return loss


# ---------------------------------------------------------------------------
# MLP backbone (used by both SL and OL)
# ---------------------------------------------------------------------------
# dropout_rate defaults to 0.0 so all existing SL code is unafected
# OL experiments pass dropout_rate > 0 for Part 3 regularization study
class SimpleMLP(nn.Module):
    """Compact MLP with optional dropout after every hidden ReLU.

    dropout_rate=0.0 by default so all existing SL code is unaffected.
    OL experiments pass dropout_rate > 0 for Part 3 regularization study.
    """

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))  # fully connected layer
            layers.append(nn.ReLU())  # activation function - ReLU works better than sigmoid here
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))  # only add dropout if rate is set
            prev = h
        layers.append(nn.Linear(prev, output_dim))  # final output layer (no activation - loss handles it)
        self.net          = nn.Sequential(*layers)
        self.dropout_rate = dropout_rate  # save so we can inspect later

    def forward(self, x):
        return self.net(x)

    # ------------------------------------------------------------------
    # Helpers for randomized-optimization layer access (OL Part 1)
    # These let RO algorithms read/write only the last few layers
    # without touching the frozen feature extractor layers
    # ------------------------------------------------------------------
    def _get_linear_layers(self):
        """Return only the Linear modules from self.net (skips ReLU/Dropout)."""
        return [m for m in self.net.children() if isinstance(m, nn.Linear)]

    def get_last_layer_params(self, num_layers=2):
        """Return flat numpy array of the last `num_layers` linear layers.
        
        Flattens weights + biases into a single 1D vector that RO
        algorythms can treat as the search space. Order is: weight, bias
        for each layer from second-to-last to last.
        """
        target = self._get_linear_layers()[-num_layers:]
        parts  = []
        for layer in target:
            parts.append(layer.weight.data.flatten())
            parts.append(layer.bias.data.flatten())
        return torch.cat(parts).cpu().numpy()

    def set_last_layer_params(self, params_vector, num_layers=2):
        """Write a flat numpy array back into the last `num_layers` linear layers.
        
        Inverse of get_last_layer_params - takes the 1D vector from the RO
        algorythm and unpacks it back into the actual weight tensors. Raises
        ValueError if the vector length doesnt match (catches bugs early).
        """
        target = self._get_linear_layers()[-num_layers:]
        idx    = 0
        for layer in target:
            w_size = layer.weight.numel()
            b_size = layer.bias.numel()
            layer.weight.data = torch.tensor(
                params_vector[idx:idx + w_size], dtype=torch.float32
            ).reshape(layer.weight.shape)
            idx += w_size
            layer.bias.data = torch.tensor(
                params_vector[idx:idx + b_size], dtype=torch.float32
            ).reshape(layer.bias.shape)
            idx += b_size
        if idx != len(params_vector):
            raise ValueError(
                f"Parameter count mismatch: consumed {idx}, "
                f"got {len(params_vector)}")


# ---------------------------------------------------------------------------
# Freezing helper (OL Part 1)
# ---------------------------------------------------------------------------
# Freeze everything except last N linear layers so RO only optimizes
# the top of the network. Speeds up search dramaticaly since the
# paramater space is much smaller
def freeze_all_but_last(model, num_layers=2):
    """Freeze all parameters except the last `num_layers` linear layers."""
    for param in model.parameters():
        param.requires_grad = False  # freeze everyting first
    for layer in model._get_linear_layers()[-num_layers:]:
        layer.weight.requires_grad = True  # then unfreeze just the last layers
        layer.bias.requires_grad   = True


# ---------------------------------------------------------------------------
# Core training loop (used by both SL tuning and OL warm-start)
# ---------------------------------------------------------------------------
# Note: switched to Adam here from SGD used in SL - better convergance
# for the OL experiments. SGD is still used in sklearn MLP tuner below
def train_pytorch_model(X_train, y_train, X_val, y_val,
                        input_dim, hidden_dims, output_dim,
                        lr=0.01, weight_decay=0.0, epochs=100,
                        batch_size=64, early_stopping=5,
                        dropout_rate=0.0, verbose=False):
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use GPU if availble
    model     = SimpleMLP(input_dim, hidden_dims, output_dim,
                          dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()  # standard clasification loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # Adam with bias correction (defualt)

    # Convert numpy arrays to PyTorch tensors - needs to be float32 not float64
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size, shuffle=True)  # shuffle every epoch for better generalization

    best_val_loss = float('inf')
    best_epoch    = 0
    best_state    = None  # stores weights at best validation loss
    history       = {'train_loss': [], 'val_loss': [],
                     'train_acc':  [], 'val_acc':  []}

    for epoch in range(epochs):
        model.train()  # set to training mode (enables dropout etc)
        running_loss = running_n = correct = total = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()  # clear gradients from last step
            out  = model(bx)
            loss = criterion(out, by)
            loss.backward()  # backprop through the whole network
            optimizer.step()  # update weights
            running_loss += loss.item() * bx.size(0)
            running_n    += bx.size(0)
            _, pred = torch.max(out, 1)
            total   += by.size(0)
            correct += (pred == by).sum().item()

        history['train_loss'].append(running_loss / running_n)
        history['train_acc'].append(correct / total)

        # Validation phase - no gradient needed here
        model.eval()  # set to eval mode (disables dropout)
        with torch.no_grad():
            val_out  = model(X_val_t.to(device))
            val_loss = criterion(val_out, y_val_t.to(device)).item()
            _, vpred = torch.max(val_out, 1)
            val_acc  = (vpred.cpu() == y_val_t).float().mean().item()
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Early stopping - save best weights and stop if no improvment
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            best_state    = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}  # deep copy the weights
        elif epoch - best_epoch >= early_stopping:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    # Restore best weights before returning
    if best_state:
        model.load_state_dict(best_state)
    return model.to(device), history


# Tune PyTorch NN over learning rate, weight decay, and architechture
# Returns best model, training history, and config dict
def tune_pytorch_nn(X_train, y_train, X_val, y_val, input_dim, output_dim,
                    lr_list=None, wd_list=None, hidden_archs=None):
    if lr_list      is None: lr_list      = [0.01, 0.001]
    if wd_list      is None: wd_list      = [0, 1e-4]
    if hidden_archs is None: hidden_archs = [[100], [50, 50]]  # shallow-wide vs deep-narrow

    best_model = best_history = None
    best_score = -1
    best_config = {}

    for hidden in hidden_archs:
        for lr in lr_list:
            for wd in wd_list:
                model, history = train_pytorch_model(
                    X_train, y_train, X_val, y_val,
                    input_dim, hidden, output_dim,
                    lr=lr, weight_decay=wd, epochs=100,
                    batch_size=64, early_stopping=5)
                score = history['val_acc'][-1]  # use final val accuracy to pick best
                if score > best_score:
                    best_score   = score
                    best_model   = model
                    best_history = history
                    best_config  = {'hidden': hidden, 'lr': lr, 'wd': wd}
    return best_model, best_history, best_config


# ---------------------------------------------------------------------------
# sklearn model tuners (SL Report - unchanged from original models.py)
# ---------------------------------------------------------------------------

# Helper for cross-validation scoring
# Computes mean CV score across folds for a given estimator
def _cv_score(estimator, X_train, y_train, cv, scoring):
    scores = []
    for tr, va in cv.split(X_train, y_train):
        # Clone the estimator so we start fresh each fold
        est = estimator.__class__(**estimator.get_params())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X_train[tr], y_train[tr])
            y_pred = est.predict(X_train[va])
        # Score based on task type - binary vs multiclass
        if scoring == 'f1':
            s = evaluate_model(y_train[va], y_pred, task='binary')[1]
        elif scoring == 'f1_macro':
            s = evaluate_model(y_train[va], y_pred, task='multiclass')[1]
        else:
            s = evaluate_model(y_train[va], y_pred, task='binary')[1]
        scores.append(s)
    return np.mean(scores)


# Tune max_depth and ccp_alpha for Decision Tree
# ccp_alpha controls post-pruning strength - higher = more aggresive pruning
def tune_decision_tree(X_train, y_train, cv, scoring):
    best_params, best_score = None, -1
    for depth in [3, 5, 10, 15, 20, None]:  # shallow to deep, None = unlimited
        for alpha in np.logspace(-4, -1, 4):  # post-pruning regularization strength
            dt = DecisionTreeClassifier(
                max_depth=depth, ccp_alpha=alpha, random_state=42)
            s = _cv_score(dt, X_train, y_train, cv, scoring)
            if s > best_score:
                best_score, best_params = s, {'max_depth': depth,
                                               'ccp_alpha': alpha}
    # Retrain on full training data with best params
    m = DecisionTreeClassifier(**best_params, random_state=42)
    m.fit(X_train, y_train)
    return m


# Tune n_neighbors for kNN
# Distance weighting helps with class imbalance - closer points count more
def tune_knn(X_train, y_train, cv, scoring):
    best_k, best_score = None, -1
    for k in [3, 5, 11, 21]:  # small k = low bias high variance, large k = oposite
        s = _cv_score(
            KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1),
            X_train, y_train, cv, scoring)
        if s > best_score:
            best_score, best_k = s, k
    m = KNeighborsClassifier(n_neighbors=best_k, weights='distance', n_jobs=-1)
    m.fit(X_train, y_train)
    return m


# Ultra-fast linear SVM using SGD - O(n) complexity
# Best for large datasets like Adult (30k+ samples)
def tune_svm_sgd(X_train, y_train, cv, scoring):
    best_score, best_model = -1, None
    for alpha in [0.0001, 0.001]:  # regularization strength - smaller = less regularization
        sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=alpha,
                            learning_rate='optimal',  # adaptivly computed lr
                            max_iter=1000, tol=1e-3,
                            n_jobs=-1, early_stopping=True,
                            validation_fraction=0.1, n_iter_no_change=5,
                            random_state=42)
        s = _cv_score(sgd, X_train, y_train, cv, scoring)
        if s > best_score:
            best_score = s
            # Retrain with winning alpha on full data
            best_model = SGDClassifier(
                loss='hinge', penalty='l2', alpha=alpha,
                learning_rate='optimal', max_iter=1000, tol=1e-3,
                n_jobs=-1, early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=5, random_state=42)
            best_model.fit(X_train, y_train)
    return best_model


# Fast LinearSVC - O(n_features * n_samples)
# Better for medium-sized datasets where SGD might not converge as reliably
def tune_svm_linear_fast(X_train, y_train, cv, scoring):
    best_score, best_model = -1, None
    for C in np.logspace(-2, 2, 3):  # inverse regularization - larger C = less regularizaton
        svm = LinearSVC(C=C, loss='squared_hinge', dual=False,
                        random_state=42, max_iter=2000, tol=1e-3,
                        multi_class='ovr')  # one-vs-rest for multiclass problems
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = _cv_score(svm, X_train, y_train, cv, scoring)
        if s > best_score:
            best_score = s
            best_model = LinearSVC(C=C, loss='squared_hinge', dual=False,
                                   random_state=42, max_iter=2000, tol=1e-3,
                                   multi_class='ovr')
            best_model.fit(X_train, y_train)
    return best_model


# Auto-selects fastest linear SVM based on dataset size
# Large datasets (>10k samples) get SGD, medium gets LinearSVC
def tune_svm_linear(X_train, y_train, cv, scoring):
    if X_train.shape[0] > 10000:
        return tune_svm_sgd(X_train, y_train, cv, scoring)
    return tune_svm_linear_fast(X_train, y_train, cv, scoring)


# Optimized RBF SVM - runs in ~5-10 mins on Adult, ~20-30 secs on Wine
# Key optimizations: reduced param grid, kernel caching, relaxed tolerence
def tune_svm_rbf(X_train, y_train, cv, scoring):
    n = X_train.shape[0]
    # Agressive grid reduction for large datasets to keep runtime managable
    C_range     = [0.1, 1.0, 10.0] if n > 10000 else np.logspace(-2, 2, 3)
    gamma_range = ['scale', 'auto'] if n > 10000 else ['scale', 'auto', 0.1]
    cache_size  = 1024 if n > 10000 else 512  # kernel cache in MB - avoids recomputing
    tol         = 1e-3  # relaxed from defualt 1e-4, ~10x speedup with minimal accuracy loss
    max_iter    = 3000 if n > 10000 else 5000  # fewer iters for tuning, more for final model

    best_score, best_params = -1, None
    for C in C_range:
        for gamma in gamma_range:
            svm = SVC(kernel='rbf', C=C, gamma=gamma, probability=False,  # probability=False speeds up training
                      random_state=42, cache_size=cache_size,
                      max_iter=max_iter, tol=tol, shrinking=True,  # shrinking heuristic helps speed
                      decision_function_shape='ovr')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = _cv_score(svm, X_train, y_train, cv, scoring)
            if s > best_score:
                best_score, best_params = s, {'C': C, 'gamma': gamma}

    # Use more iterations for final model since we're not searching anymore
    final_iter = 15000 if n > 10000 else 10000
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_model = SVC(kernel='rbf', **best_params,
                         probability=(scoring == 'f1'),  # only enable if needed - slows things down
                         random_state=42, cache_size=cache_size,
                         max_iter=final_iter, tol=tol, shrinking=True,
                         decision_function_shape='ovr')
        best_model.fit(X_train, y_train)
    return best_model


# Tune sklearns MLPClassifier
# Tests different hidden layer sizes and L2 regularization strengths
# Uses SGD with no momentum to match assignement requirements
def tune_sklearn_nn(X_train, y_train, cv, scoring):
    best_score, best_model = -1, None
    for sizes in [(50,), (100,)]:  # shallow-wide architechtures only (assignment constraint)
        for alpha in [0.0001, 0.001]:  # L2 penalty - helps with overfiting
            mlp = MLPClassifier(hidden_layer_sizes=sizes, alpha=alpha,
                                learning_rate_init=0.01, max_iter=200,
                                random_state=42, solver='sgd',
                                momentum=0,  # no momentum - required by assignement
                                early_stopping=True, validation_fraction=0.1,
                                n_iter_no_change=5)  # stop if no improvment for 5 iters
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = _cv_score(mlp, X_train, y_train, cv, scoring)
            if s > best_score:
                best_score = s
                best_model = MLPClassifier(
                    hidden_layer_sizes=sizes, alpha=alpha,
                    learning_rate_init=0.01, max_iter=200,
                    random_state=42, solver='sgd', momentum=0,
                    early_stopping=True, validation_fraction=0.1,
                    n_iter_no_change=5)
                best_model.fit(X_train, y_train)
    return best_model