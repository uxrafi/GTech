#!/usr/bin/env python
"""
Part 4 (Extra Credit): Integrated best combination on Adult Income.

What this does:

- Combines the best elements from Parts 1, 2, and 3 into a single training recipe
- Trains three conditions for comparision: baseline Adam, best regularization, and integrated
- The integrated condition adds GA fine tuning on top of the best regularization model
- Generates a frontier plot showing test F1 vs total compute for all three conditions
- Generates a final comparison panel (3 subplots: F1, compute, wall-clock time)
- Explicitly accepts or rejects hypotheses with quantitative evidence

The goal is to test whether combining the best optimizer, best regularization,
and RO fine tuning gives additive gains — or whether the interventions interfere
with each other when stacked together.
"""

##########################

"""
Assignment Requirements Covered:

- Part 4 extra credit integration of all three components
- Optimizer: standard Adam with best Part 2 hyperparameters (not retuned)
- Regularization: best combo from Part 3 loaded from JSON (dropout + label smoothing + L2 + ES)
- RO fine tuning: GA applied to last 2 linear layers, capped at 500 evals (10% of Part 1 budget)
- Parameter cap: same <=50k trainable parameter constraint as Part 1
- No new hyperparameter sweeps — uses values from Parts 1-3 directly
- Frontier plot: test F1 vs total compute for all three conditions with IQR error bars
- Final comparison panel: 3-subplot figure covering F1, compute cost, and wall-clock time
- Hypothesis resolution: explicitly states whether each condition improved vs baseline
- Results saved to JSON
- Compute accounting: gradient evals and function evals tracked and reported separately
- Seeds matched: same seeds and initial weights as Parts 1-3 for fair comparision
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt

from data_loader import load_adult
from models import SimpleMLP, freeze_all_but_last
from ro_optimizers import GeneticAlgorithm
from utils import set_seeds, evaluate_model, median_iqr
from paths import OUTPUT_DIR, FIGURES_DIR, RANDOM_STATE
from part3_regularization import LabelSmoothingCrossEntropy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_best_adult_config():
    """Load best hyperparameters from SL results JSON if available.
    Part 4 uses these settings for Adam and does not retune them."""
    try:
        with open(OUTPUT_DIR / 'adult_results.json') as f:
            return json.load(f)['NN_pytorch']['best_params']
    except (FileNotFoundError, KeyError):
        print("Warning: adult_results.json not found. Using defaults.")
        return {'hidden': [100], 'lr': 0.001, 'wd': 1e-4}


def load_part3_best_values():
    """Load best regularization values found in Part 3 sweep.
    Used directly in Part 4 — no new sweeps allowed per assignment rules.
    Falls back to conservative defaults if Part 3 JSON is missing."""
    try:
        with open(OUTPUT_DIR / 'part3_regularization_results.json') as f:
            return json.load(f)['best_values']
    except (FileNotFoundError, KeyError):
        print("Warning: part3 results not found. Using defaults.")
        return {'best_l2': 1e-4, 'best_dropout': 0.3,
                'best_label_smoothing': 0.1, 'best_input_noise': 0.01,
                'early_stop_patience': 5}


def extract_tensors(loaders):
    """Pull raw numpy arrays out of the DataLoader tensors."""
    tr, va, te = loaders[:3]
    def np_(l): X, y = l.dataset.tensors; return X.numpy(), y.numpy()
    return *np_(tr), *np_(va), *np_(te)


def make_loader(X, y, batch_size, shuffle):
    """Build a DataLoader from numpy arrays.
    shuffle=True for training, shuffle=False for val/test."""
    ds = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_full(model, criterion, optimizer, X_train, y_train, val_loader,
               num_epochs, early_stop_patience, device):
    """Core training loop with optional early stopping.

    Returns the model at its best validation loss checkpoint and the total
    gradient eval count. Gradient evals counted as one per batch backward
    pass — used for compute accounting in the frontier plot.
    Always uses standard CrossEntropyLoss for validation — not label smoothing.
    """
    tl = make_loader(X_train, y_train, 64, True)
    best_loss  = float('inf')
    best_state = None
    pat        = 0
    grad_evals = 0

    for _ in range(num_epochs):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
            grad_evals += 1

        model.eval()
        tl2 = tn = 0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                tl2 += nn.CrossEntropyLoss()(model(xv), yv).item() * xv.size(0)
                tn  += xv.size(0)
        vl = tl2 / tn

        if vl < best_loss:
            best_loss  = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if early_stop_patience and pat >= early_stop_patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model.to(device), grad_evals


def test_f1(model, X_test, y_test, device):
    """Evaluate test F1. Always runs in eval mode so dropout is off."""
    model.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        _, pred = torch.max(model(Xt), 1)
    _, f1 = evaluate_model(y_test, pred.cpu().numpy(), task='binary')
    return f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_composition(seeds=(42, 43, 44)):
    print("\n=== Part 4: Integrated Best Combination (Extra Credit) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loaders = load_adult()
    X_train, y_train, X_val, y_val, X_test, y_test = extract_tensors(loaders)

    y_train = y_train.astype(np.int64).flatten()
    y_val   = y_val.astype(np.int64).flatten()
    y_test  = y_test.astype(np.int64).flatten()

    input_dim  = X_train.shape[1]
    output_dim = 2

    # Load Adam config from SL results — not retuned here
    cfg     = load_best_adult_config()
    hidden  = cfg['hidden']
    base_lr = cfg['lr']
    base_wd = cfg['wd']

    # Load regularization values from Part 3 — not retuned here
    p3           = load_part3_best_values()
    best_l2      = p3['best_l2']
    best_dropout = p3['best_dropout']
    best_smooth  = p3['best_label_smoothing']
    es_patience  = p3['early_stop_patience']

    print(f"Adam config: hidden={hidden}, lr={base_lr}")
    print(f"Reg config:  dropout={best_dropout}, smooth={best_smooth}, "
          f"L2={best_l2}, ES patience={es_patience}")

    NUM_EPOCHS = 30
    RO_BUDGET  = 500  # capped at 10% of Part 1 budget (5000 evals)

    val_loader = make_loader(X_val, y_val, 256, False)

    # Storage for per-seed results
    frontier = {
        'baseline_adam':       {'f1': [], 'compute': [], 'time_s': []},
        'best_reg':            {'f1': [], 'compute': [], 'time_s': []},
        'integrated_after_ro': {'f1': [], 'compute': [], 'time_s': []},
    }

    for seed in seeds:
        print(f"\n  Seed {seed}")
        set_seeds(seed)

        # Snapshot shared initial weights — all three conditions start from here
        ref_model  = SimpleMLP(input_dim, hidden, output_dim)
        init_state = copy.deepcopy(ref_model.state_dict())

        # ---- (A) Baseline Adam — no regularization --------------------------
        t0      = time.time()
        model_a = SimpleMLP(input_dim, hidden, output_dim).to(device)
        model_a.load_state_dict(copy.deepcopy(init_state))
        opt_a   = optim.Adam(model_a.parameters(), lr=base_lr, weight_decay=base_wd)
        model_a, ge_a = train_full(model_a, nn.CrossEntropyLoss(), opt_a,
                                   X_train, y_train, val_loader,
                                   NUM_EPOCHS, None, device)
        t_a  = time.time() - t0
        f1_a = test_f1(model_a, X_test, y_test, device)
        frontier['baseline_adam']['f1'].append(float(f1_a))
        frontier['baseline_adam']['compute'].append(ge_a)
        frontier['baseline_adam']['time_s'].append(t_a)
        print(f"    (A) baseline Adam: F1={f1_a:.4f}, {ge_a} grad evals, {t_a:.1f}s")

        # ---- (B) Best regularization — Adam + all four techniques -----------
        set_seeds(seed)
        t0      = time.time()
        model_b = SimpleMLP(input_dim, hidden, output_dim,
                            dropout_rate=best_dropout).to(device)
        model_b.load_state_dict(copy.deepcopy(init_state), strict=False)
        opt_b   = optim.Adam(model_b.parameters(), lr=base_lr, weight_decay=best_l2)
        crit_b  = LabelSmoothingCrossEntropy(smoothing=best_smooth)
        model_b, ge_b = train_full(model_b, crit_b, opt_b,
                                   X_train, y_train, val_loader,
                                   NUM_EPOCHS, es_patience, device)
        t_b  = time.time() - t0
        f1_b = test_f1(model_b, X_test, y_test, device)
        frontier['best_reg']['f1'].append(float(f1_b))
        frontier['best_reg']['compute'].append(ge_b)
        frontier['best_reg']['time_s'].append(t_b)
        print(f"    (B) best reg:      F1={f1_b:.4f}, {ge_b} grad evals, {t_b:.1f}s")

        # ---- (C) Integrated — condition B + GA fine tuning ------------------
        # Freeze all but last 2 layers then run GA on those weights.
        # GA budget capped at 500 evals — 10% of Part 1 budget.
        set_seeds(seed)
        freeze_all_but_last(model_b, num_layers=2)
        init_ro_params = model_b.get_last_layer_params(num_layers=2)

        n_trainable = sum(p.numel() for p in model_b.parameters()
                          if p.requires_grad)
        print(f"    RO trainable params: {n_trainable:,}")

        def ro_objective(params):
            """Val loss objective for GA. Dropout off during RO evaluation."""
            model_b.eval()
            model_b.set_last_layer_params(params, num_layers=2)
            tl = tn = 0
            with torch.no_grad():
                for xv, yv in val_loader:
                    xv, yv = xv.to(device), yv.to(device)
                    out = model_b(xv)
                    tl += nn.CrossEntropyLoss()(out, yv).item() * xv.size(0)
                    tn += xv.size(0)
            return tl / tn

        # GA with reduced population (20 vs 30 in Part 1) to fit 500 eval budget
        ga = GeneticAlgorithm(pop_size=20, mutation_rate=0.1, mutation_std=0.01,
                              crossover_rate=0.8, elitism=2,
                              max_evaluations=RO_BUDGET)
        t0_ro = time.time()
        best_ro_params, best_ro_loss, ro_evals, ro_history = ga.optimize(
            ro_objective, init_ro_params.copy(), return_history=True)
        t_ro = time.time() - t0_ro

        model_b.set_last_layer_params(best_ro_params, num_layers=2)
        f1_c          = test_f1(model_b, X_test, y_test, device)
        total_compute = ge_b + ro_evals
        total_time    = t_b + t_ro

        frontier['integrated_after_ro']['f1'].append(float(f1_c))
        frontier['integrated_after_ro']['compute'].append(total_compute)
        frontier['integrated_after_ro']['time_s'].append(total_time)
        print(f"    (C) +RO fine-tune: F1={f1_c:.4f}, "
              f"{ro_evals} RO evals, total compute={total_compute}, "
              f"{total_time:.1f}s")

    # -----------------------------------------------------------------------
    # Figure 1: Frontier plot — test F1 vs total compute
    # An ideal intervention moves up and left: higher F1, less compute.
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    styles = {
        'baseline_adam':       ('o', '#1f77b4', 'Baseline Adam (A)'),
        'best_reg':            ('s', '#2ca02c', 'Best regularization (B)'),
        'integrated_after_ro': ('^', '#d62728', 'Integrated + RO fine-tune (C)'),
    }
    for key, (marker, color, label) in styles.items():
        f1s      = frontier[key]['f1']
        coms     = frontier[key]['compute']
        med_f1,  iqr_f1  = median_iqr(f1s)
        med_com, iqr_com = median_iqr(coms)
        ax.scatter(med_com, med_f1, marker=marker, color=color, s=150,
                   zorder=5, label=f'{label}\n  F1={med_f1:.4f}±{iqr_f1:.4f}')
        ax.errorbar(med_com, med_f1,
                    xerr=iqr_com / 2, yerr=iqr_f1 / 2,
                    fmt='none', color=color, capsize=4, alpha=0.6)

    ax.set_xlabel('Total compute (gradient + function evaluations)')
    ax.set_ylabel('Test F1')
    ax.set_title('Part 4: Frontier plot — Adult (median ± IQR, 3 seeds)')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'part4_frontier.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved part4_frontier.png")

    # -----------------------------------------------------------------------
    # Figure 2: Final comparison panel
    # Panel 1: Test F1 bar chart (median + IQR)
    # Panel 2: Total compute bar chart (grad evals + func evals)
    # Panel 3: Wall-clock time bar chart
    # -----------------------------------------------------------------------
    conditions = ['baseline_adam', 'best_reg', 'integrated_after_ro']
    labels     = ['Baseline\nAdam (A)', 'Best Reg\n(B)', 'Integrated\n+RO (C)']
    bar_colors = ['#1f77b4', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    # Panel 1: Test F1
    ax = axes[0]
    meds, errs = [], []
    for cond in conditions:
        med, iqr = median_iqr(frontier[cond]['f1'])
        meds.append(med)
        errs.append(iqr / 2)
    bars = ax.bar(labels, meds, color=bar_colors,
                  yerr=errs, capsize=6, alpha=0.85)
    ax.set_ylabel('Test F1')
    ax.set_title('Test F1\n(median ± IQR, 3 seeds)')
    ax.set_ylim(max(0, min(meds) - 0.02), min(1.0, max(meds) + 0.02))
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for bar, med in zip(bars, meds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f'{med:.4f}', ha='center', va='bottom', fontsize=8)

    # Panel 2: Total compute
    ax = axes[1]
    meds_c, errs_c = [], []
    for cond in conditions:
        med, iqr = median_iqr(frontier[cond]['compute'])
        meds_c.append(med)
        errs_c.append(iqr / 2)
    bars = ax.bar(labels, meds_c, color=bar_colors,
                  yerr=errs_c, capsize=6, alpha=0.85)
    ax.set_ylabel('Total compute\n(grad + func evals)')
    ax.set_title('Compute cost\n(median ± IQR, 3 seeds)')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for bar, med in zip(bars, meds_c):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(meds_c) * 0.01,
                f'{int(med):,}', ha='center', va='bottom', fontsize=7)

    # Panel 3: Wall-clock time
    ax = axes[2]
    meds_t, errs_t = [], []
    for cond in conditions:
        med, iqr = median_iqr(frontier[cond]['time_s'])
        meds_t.append(med)
        errs_t.append(iqr / 2)
    bars = ax.bar(labels, meds_t, color=bar_colors,
                  yerr=errs_t, capsize=6, alpha=0.85)
    ax.set_ylabel('Wall-clock time (s)')
    ax.set_title('Training time\n(median ± IQR, 3 seeds)')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for bar, med in zip(bars, meds_t):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(meds_t) * 0.01,
                f'{med:.1f}s', ha='center', va='bottom', fontsize=8)

    fig.suptitle(
        'Part 4: Final comparison — Baseline Adam vs Best Regularization vs Integrated + RO\n'
        'Adult Income dataset',
        fontsize=10
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'part4_final_comparison_panel.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved part4_final_comparison_panel.png")

    # -----------------------------------------------------------------------
    # Hypothesis resolution
    # -----------------------------------------------------------------------
    med_a, iqr_a = median_iqr(frontier['baseline_adam']['f1'])
    med_b, iqr_b = median_iqr(frontier['best_reg']['f1'])
    med_c, iqr_c = median_iqr(frontier['integrated_after_ro']['f1'])

    print(f"\n  Hypothesis resolution:")
    print(f"    Baseline Adam F1          = {med_a:.4f} ± {iqr_a:.4f}")
    print(f"    Best regularization F1    = {med_b:.4f} ± {iqr_b:.4f}  "
          f"({'IMPROVED' if med_b > med_a else 'NO IMPROVEMENT'} vs baseline, "
          f"delta={med_b - med_a:+.4f})")
    print(f"    Integrated + RO fine-tune = {med_c:.4f} ± {iqr_c:.4f}  "
          f"({'IMPROVED' if med_c > med_b else 'NO IMPROVEMENT'} vs best_reg, "
          f"delta={med_c - med_b:+.4f})")
    print(f"    Overall delta (A->C)      = {med_c - med_a:+.4f}")

    # Save all results to JSON
    with open(OUTPUT_DIR / 'part4_composition_results.json', 'w') as f:
        json.dump(frontier, f, indent=2)
    print("\n  Part 4 done. Results saved.")


if __name__ == '__main__':
    run_composition()