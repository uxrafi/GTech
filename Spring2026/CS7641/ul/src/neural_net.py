"""
Neural network architecture and training loop.

What this does:

- Defines the MLP class matching the OL assignment baseline exactly
- Provides train_nn() which trains the model and returns evaluation metrics
- Records per-epoch wall clock time so we can compare training speed across
  different input representations in Steps 4 and 5

The architecture is a 2-layer MLP: input -> 100 -> n_classes with ReLU
activation and Dropout(0.3) after the hidden layer, matching the OL report
best config (hidden=[100], dropout=0.3). We use Adam optimizer with cosine
annealing LR schedule and early stopping with patience=20 on a held-out
validation loss carved from the training data (15%, stratified) — the test
set is never touched during training so early stopping cannot influence
weight selection, and final evaluation happens exactly once after training
completes.

The input layer size is the only thing that changes between experiments
(raw 12 vs PCA/ICA/RP 4 vs cluster-augmented 16 etc.).
Everything else stays the same so that comparisons are fair and any
differences in accuracy or speed can be attributed to the input space.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.paths import RANDOM_STATE

torch.manual_seed(RANDOM_STATE)

# Validation fraction carved out of training data for early stopping.
# 0.15 of the 80% training split gives ~12% of total data as val,
# keeping test set completely unseen during training.
_VAL_FRAC = 0.15


class MLP(nn.Module):
    """MLP matching the OL report baseline: input -> 100 -> n_classes."""

    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 100), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(100, n_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_nn(X_tr, y_tr, X_te, y_te, n_classes,
             lr=1e-3, batch=64, max_ep=200, patience=20):
    """
    Train MLP, return (accuracy, macro-F1, best_epoch, mean_epoch_time_sec).

    A validation split is carved from X_tr / y_tr for early stopping.
    X_te / y_te are only used for final evaluation after training completes.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Carve validation set from training data (stratified)
    X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
        X_tr, y_tr,
        test_size=_VAL_FRAC,
        stratify=y_tr,
        random_state=RANDOM_STATE,
    )

    model   = MLP(X_tr_fit.shape[1], n_classes).to(dev)
    opt     = optim.Adam(model.parameters(), lr=lr)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
    loss_fn = nn.CrossEntropyLoss()

    # Training loader (shuffled)
    Xtr_t = torch.tensor(X_tr_fit, dtype=torch.float32)
    ytr_t = torch.tensor(y_tr_fit, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t),
        batch_size=batch,
        shuffle=True,
    )

    # Validation and test tensors (not shuffled, not used for weight updates)
    Xval_t = torch.tensor(X_val,  dtype=torch.float32).to(dev)
    yval_t = torch.tensor(y_val,  dtype=torch.long).to(dev)
    Xte_t  = torch.tensor(X_te,   dtype=torch.float32).to(dev)

    best_val  = np.inf
    pat_cnt   = 0
    best_ep   = 0
    best_state = None
    ep_times  = []

    for ep in range(max_ep):
        model.train()
        t0 = time.perf_counter()

        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        ep_times.append(time.perf_counter() - t0)
        sched.step()

        # Early stopping on held-out validation loss
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xval_t), yval_t).item()

        if val_loss < best_val - 1e-4:
            best_val   = val_loss
            pat_cnt    = 0
            best_ep    = ep
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            pat_cnt += 1

        if pat_cnt >= patience:
            break

    # Restore best weights, then evaluate on test set (first and only time)
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        preds = model(Xte_t).argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_te, preds)
    f1  = f1_score(y_te, preds, average="macro", zero_division=0)
    return acc, f1, best_ep + 1, float(np.mean(ep_times))