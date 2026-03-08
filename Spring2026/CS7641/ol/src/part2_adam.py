#!/usr/bin/env python
"""
Part 2: Adam ablations on the Adult Income dataset.

What this does:

- Trains the same MLP backbone with 7 different optimizer variants
- All optimizers start from identicle initial weights so comparisons are fair
- Records validation loss curves, wall clock time, and gradient evaluations
- Measures how fast each optimizer reaches a fixed validation loss threshold
- Generates sensitivty heatmaps over learning rate vs beta1 and beta2
- Generates generalization gap figure (train vs val loss) for all optimizers
- Reports median test F1 and IQR across 3 seeds for stabillity

The goal is not to find the best optimizer but to isolate which ingredents
of Adam actually matter — momentum, bias correction, adaptive scaling, weight decay.
"""

##########################

"""
Assignment Requirements Covered:

- Part 2 Adam ablations on Adult Income dataset only (as required)
- All 7 optimizer variants: SGD, SGD+momentum, Nesterov, Adam, Adam no bias correction,
  Adam beta1=0, AdamW
- Compute fairness: matched epochs, batch size, architeture, and initial weights per seed
- Validation loss threshold ℓ=0.35 defined once and used consistenly across all optimizers
- Wall clock time and gradient evaluation counts both recorded
- Sensitivty heatmaps over (alpha, beta1) and (alpha, beta2) on coarse grids
- Stabillity bands: median and IQR over 3 seeds for all optimizer trajectories
- Generalization gap: train vs validation loss tracked and plotted per optimizer
- Results saved to JSON for use in Part 4 intergration
- AdamNoBiasCorrection is a custom optimizer since PyTorch has no flag to disable bias correction
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from data_loader import load_adult
from models import SimpleMLP, AdamNoBiasCorrection
from utils import set_seeds, evaluate_model, median_iqr
from paths import OUTPUT_DIR, FIGURES_DIR, RANDOM_STATE


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_best_adult_config():
    """Load best hyperparameters from SL results JSON if available.
    Falls back to defaults if file is missing — these match the OL report backbone."""
    try:
        with open(OUTPUT_DIR / 'adult_results.json') as f:
            return json.load(f)['NN_pytorch']['best_params']
    except (FileNotFoundError, KeyError):
        print("Warning: adult_results.json not found. Using defaults.")
        return {'hidden': [100], 'lr': 0.001, 'wd': 1e-4}


def extract_tensors(loaders):
    """Pull raw numpy arrays out of the DataLoader tensors.
    Needed because we manually build loaders with different batch sizes."""
    tr, va, te = loaders[:3]
    def np_(l): X, y = l.dataset.tensors; return X.numpy(), y.numpy()
    Xtr, ytr = np_(tr)
    Xv,  yv  = np_(va)
    Xte, yte = np_(te)
    return Xtr, ytr, Xv, yv, Xte, yte


def make_loader(X, y, batch_size, shuffle):
    """Build a DataLoader from numpy arrays.
    shuffle=True for training loaders, shuffle=False for val/test."""
    ds = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def val_loss_fn(model, loader, criterion, device):
    """Compute average cross entropy loss over a full DataLoader.
    model.eval() enforced so dropout is off during evaluation."""
    model.eval()
    total_loss = total = 0
    with torch.no_grad():
        for xv, yv in loader:
            xv, yv = xv.to(device), yv.to(device)
            out = model(xv)
            total_loss += criterion(out, yv).item() * xv.size(0)
            total += xv.size(0)
    return total_loss / total


# ---------------------------------------------------------------------------
# Optimizer name list
# ---------------------------------------------------------------------------

OPT_NAMES = [
    'SGD (no momentum)',
    'SGD with momentum',
    'Nesterov',
    'Adam (baseline)',
    'Adam (no bias correct.)',
    'Adam (β₁=0, RMSProp)',
    'AdamW',
]


# ---------------------------------------------------------------------------
# Main ablation
# ---------------------------------------------------------------------------

def run_adam_ablations(seeds=(42, 43, 44)):
    print("\n=== Part 2: Adam Ablations on Adult ===")
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
    base_wd = cfg['wd']
    print(f"Config: hidden={hidden}, lr={base_lr}, wd={base_wd}")

    NUM_EPOCHS = 30
    BATCH_SIZE = 64
    THRESHOLD  = 0.35

    val_loader  = make_loader(X_val,  y_val,  256, False)
    test_loader = make_loader(X_test, y_test, 256, False)
    criterion   = nn.CrossEntropyLoss()

    all_results = {n: {'val_losses': [], 'train_losses': [],
                       'wall_times': [], 'grad_evals_curve': [],
                       'test_f1': [], 'time_to_thresh': [],
                       'steps_to_thresh': []}
                   for n in OPT_NAMES}

    for seed in seeds:
        print(f"\n  Seed {seed}")
        set_seeds(seed)

        ref_model  = SimpleMLP(input_dim, hidden, output_dim)
        init_state = copy.deepcopy(ref_model.state_dict())

        train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)

        for opt_name in OPT_NAMES:
            print(f"    {opt_name}...", end=' ', flush=True)
            set_seeds(seed)

            model = SimpleMLP(input_dim, hidden, output_dim).to(device)
            model.load_state_dict(copy.deepcopy(init_state))
            model.train()

            if opt_name == 'SGD (no momentum)':
                optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.0,
                                      weight_decay=base_wd)
            elif opt_name == 'SGD with momentum':
                optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9,
                                      weight_decay=base_wd)
            elif opt_name == 'Nesterov':
                optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9,
                                      weight_decay=base_wd, nesterov=True)
            elif opt_name == 'Adam (baseline)':
                optimizer = optim.Adam(model.parameters(), lr=base_lr,
                                       betas=(0.9, 0.999), weight_decay=base_wd)
            elif opt_name == 'Adam (no bias correct.)':
                optimizer = AdamNoBiasCorrection(model.parameters(), lr=base_lr,
                                                 betas=(0.9, 0.999), weight_decay=base_wd)
            elif opt_name == 'Adam (β₁=0, RMSProp)':
                optimizer = optim.Adam(model.parameters(), lr=base_lr,
                                       betas=(0.0, 0.999), weight_decay=base_wd)
            elif opt_name == 'AdamW':
                optimizer = optim.AdamW(model.parameters(), lr=base_lr,
                                        betas=(0.9, 0.999), weight_decay=base_wd)

            val_losses_curve   = []
            train_losses_curve = []
            wall_curve         = []
            grad_curve         = []
            grad_evals         = 0
            thresh_steps       = None
            thresh_time        = None
            t0                 = time.time()

            for epoch in range(NUM_EPOCHS):
                model.train()
                running_loss = running_n = 0

                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    out  = model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
                    grad_evals   += 1
                    running_loss += loss.item() * xb.size(0)
                    running_n    += xb.size(0)

                train_loss_ep = running_loss / running_n
                val_loss_ep   = val_loss_fn(model, val_loader, criterion, device)
                elapsed       = time.time() - t0

                val_losses_curve.append(val_loss_ep)
                train_losses_curve.append(train_loss_ep)
                wall_curve.append(elapsed)
                grad_curve.append(grad_evals)

                if thresh_steps is None and val_loss_ep < THRESHOLD:
                    thresh_steps = grad_evals
                    thresh_time  = elapsed

                model.train()

            model.eval()
            Xte_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            with torch.no_grad():
                _, pred = torch.max(model(Xte_t), 1)
            _, f1 = evaluate_model(y_test, pred.cpu().numpy(), task='binary')

            all_results[opt_name]['val_losses'].append(val_losses_curve)
            all_results[opt_name]['train_losses'].append(train_losses_curve)
            all_results[opt_name]['wall_times'].append(wall_curve)
            all_results[opt_name]['grad_evals_curve'].append(grad_curve)
            all_results[opt_name]['test_f1'].append(float(f1))
            all_results[opt_name]['time_to_thresh'].append(thresh_time)
            all_results[opt_name]['steps_to_thresh'].append(thresh_steps)
            print(f"F1={f1:.4f}")

    # -----------------------------------------------------------------------
    # Figure 1: Stability bands — val loss vs wall clock time
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.get_cmap('tab10', len(OPT_NAMES))

    for idx, opt_name in enumerate(OPT_NAMES):
        all_times  = all_results[opt_name]['wall_times']
        all_losses = all_results[opt_name]['val_losses']
        max_t  = max(max(t) for t in all_times if t)
        tgrid  = np.linspace(0, max_t, 200)
        mat    = np.array([np.interp(tgrid, all_times[i], all_losses[i])
                           for i in range(len(seeds))])
        med    = np.median(mat, axis=0)
        q1, q3 = np.percentile(mat, 25, axis=0), np.percentile(mat, 75, axis=0)
        c = cmap(idx)
        ax.plot(tgrid, med, color=c, label=opt_name)
        ax.fill_between(tgrid, q1, q3, color=c, alpha=0.15)

    ax.axhline(THRESHOLD, color='k', linestyle='--', linewidth=0.8,
               label=f'Threshold ℓ={THRESHOLD}')
    ax.set_xlabel('Wall-clock time (s)')
    ax.set_ylabel('Validation loss')
    ax.set_title('Optimizer stability bands — Adult (median ± IQR over 3 seeds)')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True)
    fig.savefig(FIGURES_DIR / 'part2_stability_wall.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Figure 2: Stability bands — val loss vs gradient evaluations
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, opt_name in enumerate(OPT_NAMES):
        all_ge  = all_results[opt_name]['grad_evals_curve']
        all_vl  = all_results[opt_name]['val_losses']
        max_ge  = max(max(g) for g in all_ge if g)
        ggrid   = np.linspace(0, max_ge, 200)
        mat     = np.array([np.interp(ggrid, all_ge[i], all_vl[i])
                            for i in range(len(seeds))])
        med     = np.median(mat, axis=0)
        q1, q3  = np.percentile(mat, 25, axis=0), np.percentile(mat, 75, axis=0)
        c = cmap(idx)
        ax.plot(ggrid, med, color=c, label=opt_name)
        ax.fill_between(ggrid, q1, q3, color=c, alpha=0.15)

    ax.axhline(THRESHOLD, color='k', linestyle='--', linewidth=0.8,
               label=f'Threshold ℓ={THRESHOLD}')
    ax.set_xlabel('Gradient evaluations')
    ax.set_ylabel('Validation loss')
    ax.set_title('Optimizer stability bands — Adult (grad evals)')
    ax.legend(fontsize=7)
    ax.grid(True)
    fig.savefig(FIGURES_DIR / 'part2_stability_gradevals.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Figure 3: Generalization gap — train vs validation loss per optimizer
    # One subplot per optimizer, median +/- IQR over seeds, epochs on x-axis
    # -----------------------------------------------------------------------
    n_opts = len(OPT_NAMES)
    ncols  = 4
    nrows  = (n_opts + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 3),
                             sharey=False)
    axes_flat = axes.flatten()

    for idx, opt_name in enumerate(OPT_NAMES):
        ax = axes_flat[idx]
        train_mat = np.array(all_results[opt_name]['train_losses'])  # (n_seeds, n_epochs)
        val_mat   = np.array(all_results[opt_name]['val_losses'])

        epochs = np.arange(1, train_mat.shape[1] + 1)

        med_tr = np.median(train_mat, axis=0)
        med_vl = np.median(val_mat,   axis=0)
        q1_tr, q3_tr = np.percentile(train_mat, 25, axis=0), np.percentile(train_mat, 75, axis=0)
        q1_vl, q3_vl = np.percentile(val_mat,   25, axis=0), np.percentile(val_mat,   75, axis=0)

        ax.plot(epochs, med_tr, color='#1f77b4', label='Train')
        ax.fill_between(epochs, q1_tr, q3_tr, color='#1f77b4', alpha=0.15)
        ax.plot(epochs, med_vl, color='#d62728', label='Val', linestyle='--')
        ax.fill_between(epochs, q1_vl, q3_vl, color='#d62728', alpha=0.15)

        # Shade the gap between median train and val to highlight overfitting
        ax.fill_between(epochs, med_tr, med_vl,
                        where=(med_vl > med_tr),
                        color='#ff7f0e', alpha=0.10, label='Gap')

        ax.set_title(opt_name, fontsize=8)
        ax.set_xlabel('Epoch', fontsize=7)
        ax.set_ylabel('Loss', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
        ax.grid(True, linestyle='--', alpha=0.4)

    for idx in range(n_opts, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        'Generalization gap — train vs validation loss (median ± IQR, 3 seeds)\n'
        'Adult Income dataset, matched compute budget',
        fontsize=10
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'part2_generalization_gap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Heatmaps — sensitivity of Adam to learning rate vs beta1 and beta2
    # Run on 30% data subset with 5 epochs for speed
    # -----------------------------------------------------------------------
    def _heatmap_adam(lr_range, sweep_param, sweep_range, filename, xlabel, title):
        Xtr_sub, _, ytr_sub, _ = train_test_split(
            X_train, y_train, train_size=0.3, random_state=42, stratify=y_train)
        Xv_sub,  _, yv_sub,  _ = train_test_split(
            X_val, y_val, train_size=0.3, random_state=42, stratify=y_val)
        trl = make_loader(Xtr_sub, ytr_sub, 64,  True)
        vll = make_loader(Xv_sub,  yv_sub,  256, False)

        grid = np.zeros((len(lr_range), len(sweep_range)))
        for i, lr in enumerate(lr_range):
            for j, sp in enumerate(sweep_range):
                set_seeds(42)
                m  = SimpleMLP(input_dim, hidden, output_dim).to(device)
                b1 = sp if sweep_param == 'beta1' else 0.9
                b2 = sp if sweep_param == 'beta2' else 0.999
                opt = optim.Adam(m.parameters(), lr=lr, betas=(b1, b2),
                                 weight_decay=base_wd)
                for _ in range(5):
                    m.train()
                    for xb, yb in trl:
                        xb, yb = xb.to(device), yb.to(device)
                        opt.zero_grad()
                        loss = criterion(m(xb), yb)
                        loss.backward()
                        opt.step()
                grid[i, j] = val_loss_fn(m, vll, criterion, device)

        lr_labels = [f'{lr:.0e}' for lr in lr_range]
        sp_labels  = [f'{v}' for v in sweep_range]
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(grid, xticklabels=sp_labels, yticklabels=lr_labels,
                    annot=True, fmt='.3f', cmap='viridis',
                    cbar_kws={'label': 'Val loss (5 epochs, 30% data)'}, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Learning rate α')
        ax.set_title(title)
        fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return grid

    lr_range    = [1e-4, 3e-4, 1e-3, 3e-3]
    beta1_range = [0.8, 0.9, 0.95, 0.99]
    beta2_range = [0.9, 0.99, 0.999, 0.9999]

    _heatmap_adam(lr_range, 'beta1', beta1_range,
                  'part2_heatmap_alpha_beta1.png', 'β₁',
                  'Adam sensitivity (α, β₁) — Adult')
    _heatmap_adam(lr_range, 'beta2', beta2_range,
                  'part2_heatmap_alpha_beta2.png', 'β₂',
                  'Adam sensitivity (α, β₂) — Adult')

    # -----------------------------------------------------------------------
    # Summary table — median F1, time to threshold, steps to threshold
    # -----------------------------------------------------------------------
    print("\n  Summary (median over seeds):")
    summary = {}
    for opt_name in OPT_NAMES:
        med_f1, iqr_f1 = median_iqr(all_results[opt_name]['test_f1'])
        t2t  = [v for v in all_results[opt_name]['time_to_thresh']  if v is not None]
        s2t  = [v for v in all_results[opt_name]['steps_to_thresh'] if v is not None]
        med_t2t = float(np.median(t2t)) if t2t else None
        med_s2t = int(np.median(s2t))   if s2t else None
        summary[opt_name] = {'median_f1': med_f1, 'iqr_f1': iqr_f1,
                             'median_time_to_thresh': med_t2t,
                             'median_steps_to_thresh': med_s2t}
        reached = f"{med_t2t:.2f}s / {med_s2t} steps" if med_t2t else "never reached"
        print(f"    {opt_name:<30} F1={med_f1:.4f}±{iqr_f1:.4f}  thresh@{reached}")

    # Save full results to JSON for use in Part 4
    with open(OUTPUT_DIR / 'part2_adam_results.json', 'w') as f:
        json.dump({'per_optimizer': all_results, 'summary': summary}, f, indent=2)
    print("\n  Part 2 done.")


if __name__ == '__main__':
    run_adam_ablations()