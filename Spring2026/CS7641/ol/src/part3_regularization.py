#!/usr/bin/env python
"""
Part 3: Regularization study on the Adult Income dataset.

What this does:

- Tests four regularization techniques on the same MLP backbone from the SL report
- Uses standard Adam with best Part 2 hyperparameters — optimizer is not changed here
- Runs a small sweep to pick the best value for each regularizer before full evaluation
- Evaluates each technique individualy and then as a combined best combo
- Reports median test F1 and IQR across 3 seeds for stabillity
- Generates a bar chart of test F1 across all regularization configs

The goal is to measure how much each regularization technique moves the test metric
relative to the unregularized baseline — not to find the best possible model.
"""

##########################

"""
Assignment Requirements Covered:

- Part 3 regularization study on Adult Income only (as required)
- All four required techniques: L2 weight decay, early stopping, dropout, noise regularization
- Optimizer constraint: standard Adam with best Part 2 hyperparameters — Adam not retuned here
- Backbone constraint: same SimpleMLP as SL report — only regularization modules added/removed
- Candidate sweep: small grid search to justify chosen regularizer values before full eval
- Dropout placement documented: inserted after every hidden ReLU via dropout_rate arg
- Input noise is training only — val and test always use clean data (no leakage)
- Label smoothing is training only — uses custom LabelSmoothingCrossEntropy class
- Best combo: all four techniques stacked with dropout capped at 0.1 to avoid over-regularization
- Results and best values saved to JSON for use in Part 4 intergration
- Compute budget matched: all configs use same NUM_EPOCHS and BATCH_SIZE
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data_loader import load_adult
from models import SimpleMLP
from utils import set_seeds, evaluate_model, median_iqr
from paths import OUTPUT_DIR, FIGURES_DIR, RANDOM_STATE


# ---------------------------------------------------------------------------
# Label smoothing loss
# ---------------------------------------------------------------------------

class LabelSmoothingCrossEntropy(nn.Module):
    """Uniform label smoothing cross entropy loss.

    Instead of training the model to output probability 1.0 for the correct
    class, label smoothing targets (1 - smoothing) for the correct class and
    smoothing/(n-1) for all other classes. Only applied during training —
    val and test always use standard CrossEntropyLoss.
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n = pred.size(1)
        with torch.no_grad():
            smooth = torch.full_like(pred, self.smoothing / (n - 1))
            smooth.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        log_p = torch.log_softmax(pred, dim=1)
        return -(smooth * log_p).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Noisy dataset wrapper
# ---------------------------------------------------------------------------

class NoisyDataset(torch.utils.data.Dataset):
    """Wraps a dataset and adds fresh Gaussian noise to inputs each access.

    Noise is sampled independently every time __getitem__ is called so every
    epoch sees slightly different noisy versions of the same sample.
    Val and test loaders never use this wrapper — noise is training only.
    """
    def __init__(self, X, y, noise_std):
        self.X         = torch.tensor(X, dtype=torch.float32)
        self.y         = torch.tensor(y, dtype=torch.long)
        self.noise_std = noise_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx] + self.noise_std * torch.randn_like(self.X[idx])
        return x, self.y[idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_best_adult_config():
    """Load best hyperparameters from SL results JSON if available.
    Part 3 uses Adam with these settings and does NOT retune the optimizer."""
    try:
        with open(OUTPUT_DIR / 'adult_results.json') as f:
            return json.load(f)['NN_pytorch']['best_params']
    except (FileNotFoundError, KeyError):
        print("Warning: adult_results.json not found. Using defaults.")
        return {'hidden': [100], 'lr': 0.001, 'wd': 1e-4}


def extract_tensors(loaders):
    """Pull raw numpy arrays out of the DataLoader tensors."""
    tr, va, te = loaders[:3]
    def np_(l):
        X, y = l.dataset.tensors
        return X.numpy(), y.numpy()
    return *np_(tr), *np_(va), *np_(te)


def make_loader(X, y, batch_size, shuffle, noisy=False, noise_std=0.0):
    """Build a DataLoader from numpy arrays.
    noisy=True wraps the dataset with NoisyDataset for input noise regularization.
    Val and test loaders always pass noisy=False to keep evaluation on clean data.
    """
    if noisy:
        ds = NoisyDataset(X, y, noise_std)
    else:
        ds = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(model, criterion, optimizer, train_loader, val_loader,
                num_epochs, early_stop_patience, device):
    """Core training loop with optional early stopping.

    Tracks best validation loss and saves model state at that point.
    Returns model loaded with weights from best validation epoch.
    Always uses standard CrossEntropyLoss for validation — not label smoothing.
    """
    best_val_loss  = float('inf')
    best_state     = None
    patience_count = 0

    for epoch in range(num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        total_loss = total = 0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                out = model(xv)
                total_loss += nn.CrossEntropyLoss()(out, yv).item() * xv.size(0)
                total      += xv.size(0)
        val_loss = total_loss / total

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if early_stop_patience and patience_count >= early_stop_patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device)
    return model


def test_f1(model, X_test, y_test, device):
    """Evaluate test F1. Always runs in eval mode so dropout is off."""
    model.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        _, pred = torch.max(model(Xt), 1)
    _, f1 = evaluate_model(y_test, pred.cpu().numpy(), task='binary')
    return f1


# ---------------------------------------------------------------------------
# Small sweep to pick best regularization value
# ---------------------------------------------------------------------------

def sweep_val(X_train, y_train, X_val, y_val, device,
              input_dim, hidden, output_dim, base_lr, base_wd,
              reg_name, candidates, sweep_epochs=20):
    """For each candidate value, train a fresh model and return the candidate
    with the lowest validation loss.

    sweep_epochs=20 so dropout models have enough steps to converge —
    shorter sweeps unfairly penalize dropout since it needs more time to stabilize.
    Only runs with seed=42 — this is a selection sweep not a stability evaluation.
    """
    best_val, best_loss = None, float('inf')
    val_loader = make_loader(X_val, y_val, 256, False)

    for val in candidates:
        set_seeds(42)
        dr    = val if reg_name == 'dropout' else 0.0
        model = SimpleMLP(input_dim, hidden, output_dim, dropout_rate=dr).to(device)

        if reg_name == 'label_smoothing':
            criterion = LabelSmoothingCrossEntropy(smoothing=val)
        else:
            criterion = nn.CrossEntropyLoss()

        wd        = val if reg_name == 'l2' else base_wd
        optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=wd)

        if reg_name == 'input_noise':
            train_loader = make_loader(X_train, y_train, 64, True,
                                       noisy=True, noise_std=val)
        else:
            train_loader = make_loader(X_train, y_train, 64, True)

        best_epoch_loss = float('inf')
        for _ in range(sweep_epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                criterion(model(xb), yb).backward()
                optimizer.step()

            model.eval()
            tl = tn = 0
            with torch.no_grad():
                for xv, yv in val_loader:
                    xv, yv = xv.to(device), yv.to(device)
                    tl += nn.CrossEntropyLoss()(model(xv), yv).item() * xv.size(0)
                    tn += xv.size(0)
            ep_loss = tl / tn
            if ep_loss < best_epoch_loss:
                best_epoch_loss = ep_loss

        if best_epoch_loss < best_loss:
            best_loss = best_epoch_loss
            best_val  = val

    return best_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_regularization(seeds=(42, 43, 44)):
    print("\n=== Part 3: Regularization Study on Adult ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loaders = load_adult()
    X_train, y_train, X_val, y_val, X_test, y_test = extract_tensors(loaders)

    y_train = y_train.astype(np.int64).flatten()
    y_val   = y_val.astype(np.int64).flatten()
    y_test  = y_test.astype(np.int64).flatten()

    input_dim  = X_train.shape[1]
    output_dim = 2

    cfg     = load_best_adult_config()
    hidden  = cfg['hidden']
    base_lr = cfg['lr']
    base_wd = cfg.get('wd', 0)
    print(f"Config: hidden={hidden}, lr={base_lr}, wd={base_wd}")

    NUM_EPOCHS  = 30
    BATCH_SIZE  = 64
    ES_PATIENCE = 5

    # Candidate values for each regularizer
    reg_candidates = {
        'l2':              [1e-5, 1e-4, 1e-3],
        'dropout':         [0.05, 0.1, 0.2, 0.3],
        'label_smoothing': [0.05, 0.1, 0.2],
        'input_noise':     [0.001, 0.01, 0.05],
    }

    SWEEP_EPOCHS = 20

    # Run sweeps to find best value for each regularizer
    print("  Sweeping regularizer candidates...")
    best_l2     = sweep_val(X_train, y_train, X_val, y_val, device,
                            input_dim, hidden, output_dim, base_lr, base_wd,
                            'l2', reg_candidates['l2'], sweep_epochs=SWEEP_EPOCHS)
    best_drop   = sweep_val(X_train, y_train, X_val, y_val, device,
                            input_dim, hidden, output_dim, base_lr, base_wd,
                            'dropout', reg_candidates['dropout'], sweep_epochs=SWEEP_EPOCHS)
    best_smooth = sweep_val(X_train, y_train, X_val, y_val, device,
                            input_dim, hidden, output_dim, base_lr, base_wd,
                            'label_smoothing', reg_candidates['label_smoothing'],
                            sweep_epochs=SWEEP_EPOCHS)
    best_noise  = sweep_val(X_train, y_train, X_val, y_val, device,
                            input_dim, hidden, output_dim, base_lr, base_wd,
                            'input_noise', reg_candidates['input_noise'],
                            sweep_epochs=SWEEP_EPOCHS)

    print(f"  Best → L2={best_l2}, dropout={best_drop}, "
          f"label_smooth={best_smooth}, input_noise={best_noise}")

    # -----------------------------------------------------------------------
    # Full multi-seed evaluation of all regularization configurations
    # -----------------------------------------------------------------------
    configs = ['baseline', 'l2', 'early_stopping', 'dropout',
               'label_smoothing', 'input_noise', 'best_combo']
    results = {c: [] for c in configs}

    val_loader = make_loader(X_val, y_val, 256, False)

    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")
        set_seeds(seed)

        def _run(model, criterion, optimizer, noisy=False, noise_std=0.0,
                 early_stop=None):
            tl = make_loader(X_train, y_train, BATCH_SIZE, True,
                             noisy=noisy, noise_std=noise_std)
            return train_model(model, criterion, optimizer, tl, val_loader,
                               NUM_EPOCHS, early_stop, device)

        # Baseline — no regularization, standard Adam
        set_seeds(seed)
        m = SimpleMLP(input_dim, hidden, output_dim).to(device)
        m = _run(m, nn.CrossEntropyLoss(),
                 optim.Adam(m.parameters(), lr=base_lr, weight_decay=base_wd))
        results['baseline'].append(test_f1(m, X_test, y_test, device))

        # L2 — weight decay via Adam weight_decay argument
        set_seeds(seed)
        m = SimpleMLP(input_dim, hidden, output_dim).to(device)
        m = _run(m, nn.CrossEntropyLoss(),
                 optim.Adam(m.parameters(), lr=base_lr, weight_decay=best_l2))
        results['l2'].append(test_f1(m, X_test, y_test, device))

        # Early stopping — stop when val loss stops improving for ES_PATIENCE epochs
        set_seeds(seed)
        m = SimpleMLP(input_dim, hidden, output_dim).to(device)
        m = _run(m, nn.CrossEntropyLoss(),
                 optim.Adam(m.parameters(), lr=base_lr, weight_decay=base_wd),
                 early_stop=ES_PATIENCE)
        results['early_stopping'].append(test_f1(m, X_test, y_test, device))

        # Dropout — inserted after every hidden ReLU via dropout_rate argument
        set_seeds(seed)
        m = SimpleMLP(input_dim, hidden, output_dim, dropout_rate=best_drop).to(device)
        m = _run(m, nn.CrossEntropyLoss(),
                 optim.Adam(m.parameters(), lr=base_lr, weight_decay=base_wd))
        results['dropout'].append(test_f1(m, X_test, y_test, device))

        # Label smoothing — custom loss applied during training only
        set_seeds(seed)
        m = SimpleMLP(input_dim, hidden, output_dim).to(device)
        m = _run(m, LabelSmoothingCrossEntropy(smoothing=best_smooth),
                 optim.Adam(m.parameters(), lr=base_lr, weight_decay=base_wd))
        results['label_smoothing'].append(test_f1(m, X_test, y_test, device))

        # Input noise — Gaussian noise added to inputs during training only
        # val and test always use clean data — no leakage
        set_seeds(seed)
        m = SimpleMLP(input_dim, hidden, output_dim).to(device)
        m = _run(m, nn.CrossEntropyLoss(),
                 optim.Adam(m.parameters(), lr=base_lr, weight_decay=base_wd),
                 noisy=True, noise_std=best_noise)
        results['input_noise'].append(test_f1(m, X_test, y_test, device))

        # Best combo — all four techniques stacked together
        # Dropout capped at 0.1 when stacking to avoid over-regularization —
        # hidden=[100] is small so aggressive dropout removes too much capacity
        combo_dropout = min(best_drop, 0.1)
        set_seeds(seed)
        m = SimpleMLP(input_dim, hidden, output_dim,
                      dropout_rate=combo_dropout).to(device)
        m = _run(m, LabelSmoothingCrossEntropy(smoothing=best_smooth),
                 optim.Adam(m.parameters(), lr=base_lr, weight_decay=best_l2),
                 early_stop=ES_PATIENCE)
        results['best_combo'].append(test_f1(m, X_test, y_test, device))

        # Print per-seed summary showing delta vs baseline
        base_f1 = results['baseline'][-1]
        for c in configs:
            f1    = results[c][-1]
            delta = f1 - base_f1
            sign  = '+' if delta >= 0 else ''
            print(f"    {c:<20} F1={f1:.4f}  ({sign}{delta:.4f} vs baseline)")

    # -----------------------------------------------------------------------
    # Aggregate results
    # -----------------------------------------------------------------------
    summary = {}
    print("\n  Final summary (median ± IQR):")
    for c in configs:
        med, iqr = median_iqr(results[c])
        summary[c] = {'median_f1': float(med), 'iqr_f1': float(iqr)}
        print(f"    {c:<20} {med:.4f} ± {iqr:.4f}")

    # -----------------------------------------------------------------------
    # Figure: bar chart of test F1 across all regularization configs
    # Grey = baseline, blue = individual regularizers, red = best combo
    # Error bars show IQR across seeds
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    meds   = [summary[c]['median_f1'] for c in configs]
    errors = [summary[c]['iqr_f1']    for c in configs]
    colors = ['#7f7f7f'] + ['#1f77b4'] * (len(configs) - 2) + ['#d62728']

    bars = ax.bar(configs, meds, yerr=errors, capsize=5, color=colors, alpha=0.85)

    # Annotate bars with exact median values
    for bar, med in zip(bars, meds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f'{med:.4f}', ha='center', va='bottom', fontsize=8)

    # Draw a horizontal line at baseline median for easy comparison
    baseline_med = summary['baseline']['median_f1']
    ax.axhline(baseline_med, color='#7f7f7f', linestyle='--',
               linewidth=1.0, label=f'Baseline F1={baseline_med:.4f}')

    ax.set_ylabel('Test F1 (median ± IQR)')
    ax.set_title('Part 3: Regularization sweep — Adult Income\n'
                 '(standard Adam, matched compute, 3 seeds)')
    ax.set_xticklabels(configs, rotation=30, ha='right')
    ax.set_ylim(max(0, min(meds) - 0.05), min(1.0, max(meds) + 0.05))
    ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'part3_reg_sweep.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved part3_reg_sweep.png")

    # -----------------------------------------------------------------------
    # Save best values to JSON for use in Part 4
    # -----------------------------------------------------------------------
    best_vals = {
        'best_l2':              best_l2,
        'best_dropout':         best_drop,
        'best_label_smoothing': best_smooth,
        'best_input_noise':     best_noise,
        'early_stop_patience':  ES_PATIENCE,
        'combo_dropout':        min(best_drop, 0.1),
    }
    with open(OUTPUT_DIR / 'part3_regularization_results.json', 'w') as f:
        json.dump({'raw': results, 'summary': summary,
                   'best_values': best_vals}, f, indent=2)
    print("\n  Part 3 done. Results saved.")

    return best_vals


if __name__ == '__main__':
    run_regularization()