#!/usr/bin/env python
"""
Part 1: Randomized Optimization on last layers of MLP (Adult + Wine).

What this does:
- Freezes all but last 2 linear layers of the pre-trained MLP backbone
- Runs RHC, SA, and GA on the frozen model's last-layer parameters
- Plots best-so-far objective vs function evaluations for each algorithm
- Reports final test F1 vs SL baseline across 3 seeds for both datasets
- Generates one RO progress figure per dataset (all 3 algorithms overlaid)
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from data_loader import load_adult, load_wine
from models import SimpleMLP, freeze_all_but_last
from ro_optimizers import RandomizedHillClimbing, SimulatedAnnealing, GeneticAlgorithm
from utils import set_seeds, evaluate_model, median_iqr
from paths import OUTPUT_DIR, FIGURES_DIR, RANDOM_STATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_best_config(dataset):
    """Load best SL hyperparams for given dataset, fall back to defaults."""
    key = 'NN_pytorch'
    defaults = {
        'adult': {'hidden': [100], 'lr': 0.001, 'wd': 1e-4},
        'wine':  {'hidden': [100], 'lr': 0.001, 'wd': 1e-4},
    }
    try:
        fname = OUTPUT_DIR / f'{dataset}_results.json'
        with open(fname) as f:
            return json.load(f)[key]['best_params']
    except (FileNotFoundError, KeyError):
        print(f"Warning: {dataset}_results.json not found. Using defaults.")
        return defaults[dataset]


def extract_tensors(loaders):
    """Pull numpy arrays from DataLoader dataset tensors."""
    def np_(l):
        X, y = l.dataset.tensors
        return X.numpy(), y.numpy()
    tr, va, te = loaders[0], loaders[1], loaders[2]
    return *np_(tr), *np_(va), *np_(te)


def make_val_objective(model, X_val, y_val, device):
    """
    Return a closure that evaluates validation cross-entropy loss
    given a flat parameter vector. model.eval() is enforced so dropout
    is off and results are deterministic — required for fair RO evaluation.
    """
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val, dtype=torch.long).to(device)
    criterion = nn.CrossEntropyLoss()

    def objective(params):
        model.eval()  # dropout off — critical for deterministic RO evaluation
        model.set_last_layer_params(params, num_layers=2)
        with torch.no_grad():
            loss = criterion(model(Xv), yv).item()
        return loss

    return objective


def pretrain_model(X_train, y_train, X_val, y_val,
                   input_dim, hidden, output_dim, cfg, device):
    """
    Quick Adam warm-start so RO starts from a reasonable point rather
    than random weights. Uses 20 epochs — enough to get a good initialisation
    without burning the RO budget on gradient descent.
    """
    from models import train_pytorch_model
    model, _ = train_pytorch_model(
        X_train, y_train, X_val, y_val,
        input_dim, hidden, output_dim,
        lr=cfg['lr'], weight_decay=cfg.get('wd', 0),
        epochs=20, batch_size=64, early_stopping=5)
    return model


def evaluate_test_f1(model, X_test, y_test, device, task='binary'):
    """Evaluate test F1 on a trained model. Dropout off during evaluation."""
    model.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        _, pred = torch.max(model(Xt), 1)
    _, f1 = evaluate_model(y_test, pred.cpu().numpy(), task=task)
    return float(f1)


# ---------------------------------------------------------------------------
# Per-dataset RO experiment
# ---------------------------------------------------------------------------

RO_BUDGET = 5000  # total function evaluation budget per algorithm per seed

# RO algorithm configs — disclosed as required by assignment
RO_CONFIGS = {
    'RHC': {
        'cls': RandomizedHillClimbing,
        'kwargs': dict(step_size=0.01, restarts=3,
                       max_evaluations=RO_BUDGET, plateau_patience=500),
        'color': '#1f77b4',
    },
    'SA': {
        'cls': SimulatedAnnealing,
        'kwargs': dict(initial_temp=1.0, decay=0.995, step_size=0.01,
                       max_evaluations=RO_BUDGET),
        'color': '#ff7f0e',
    },
    'GA': {
        'cls': GeneticAlgorithm,
        'kwargs': dict(pop_size=30, mutation_rate=0.1, mutation_std=0.01,
                       crossover_rate=0.8, elitism=2,
                       max_evaluations=RO_BUDGET),
        'color': '#2ca02c',
    },
}


def run_dataset(dataset_name, loaders, output_dim, task, seeds=(42, 43, 44)):
    """
    Run RHC, SA, GA on the last 2 layers of the MLP for one dataset.
    Returns per-algorithm results dict and saves a progress curve figure.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = load_best_config(dataset_name)
    hidden = cfg['hidden']

    # Extract numpy arrays from loaders
    if dataset_name == 'wine':
        # load_wine returns 5-tuple: tr, va, te, preprocessor, n_classes
        X_train, y_train, X_val, y_val, X_test, y_test = extract_tensors(loaders)
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = extract_tensors(loaders)

    y_train = y_train.astype(np.int64).flatten()
    y_val   = y_val.astype(np.int64).flatten()
    y_test  = y_test.astype(np.int64).flatten()

    input_dim = X_train.shape[1]

    # Storage: algo -> list of (history, final_f1) across seeds
    results = {name: {'histories': [], 'test_f1': []} for name in RO_CONFIGS}

    for seed in seeds:
        print(f"  [{dataset_name}] Seed {seed}")
        set_seeds(seed)

        # Warm-start with Adam so RO optimises from a good basin
        base_model = pretrain_model(
            X_train, y_train, X_val, y_val,
            input_dim, hidden, output_dim, cfg, device)

        # Freeze all but last 2 linear layers
        freeze_all_but_last(base_model, num_layers=2)
        n_trainable = sum(p.numel() for p in base_model.parameters()
                          if p.requires_grad)
        print(f"    Trainable params for RO: {n_trainable:,}")  # must be <=50k

        init_params = base_model.get_last_layer_params(num_layers=2)
        objective   = make_val_objective(base_model, X_val, y_val, device)

        for algo_name, algo_cfg in RO_CONFIGS.items():
            print(f"    {algo_name}...", end=' ', flush=True)
            set_seeds(seed)  # same seed per algo for fair comparison

            # Fresh copy of initial params — all algos start from same point
            p0 = init_params.copy()
            optimizer = algo_cfg['cls'](**algo_cfg['kwargs'])
            best_p, best_loss, n_evals, history = optimizer.optimize(
                objective, p0, return_history=True)

            # Load best params and evaluate on test set
            base_model.set_last_layer_params(best_p, num_layers=2)
            f1 = evaluate_test_f1(base_model, X_test, y_test, device, task=task)

            results[algo_name]['histories'].append(history)
            results[algo_name]['test_f1'].append(f1)
            print(f"F1={f1:.4f}  ({n_evals} evals)")

    # -------------------------------------------------------------------
    # Figure: best-so-far objective vs function evaluations
    # One figure per dataset, all 3 algorithms overlaid, median +/- IQR
    # Operator settings documented in title/caption as required
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))

    for algo_name, algo_cfg in RO_CONFIGS.items():
        histories = results[algo_name]['histories']
        # Interpolate all seeds onto a common eval grid
        max_evals = max(h[-1][0] for h in histories)
        egrid = np.linspace(1, max_evals, 300)
        mat = np.array([
            np.interp(egrid,
                      [pt[0] for pt in h],
                      [pt[1] for pt in h])
            for h in histories
        ])
        med    = np.median(mat, axis=0)
        q1, q3 = np.percentile(mat, 25, axis=0), np.percentile(mat, 75, axis=0)
        c = algo_cfg['color']

        # Build label with key operator settings for caption transparency
        if algo_name == 'RHC':
            lbl = f"RHC (step=0.01, restarts=3, plateau=500)"
        elif algo_name == 'SA':
            lbl = f"SA (T₀=1.0, decay=0.995, step=0.01)"
        else:
            lbl = f"GA (pop=30, mut=0.1/0.01, cx=0.8, elitism=2)"

        ax.plot(egrid, med, color=c, label=lbl)
        ax.fill_between(egrid, q1, q3, color=c, alpha=0.15)

    ax.set_xlabel('Function evaluations')
    ax.set_ylabel('Best-so-far validation loss')
    dname = dataset_name.capitalize()
    ax.set_title(
        f'Part 1: RO progress — {dname} '
        f'(median ± IQR, {len(seeds)} seeds, budget={RO_BUDGET})\n'
        f'Last 2 linear layers unfrozen; warm-started from 20-epoch Adam'
    )
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fname = FIGURES_DIR / f'part1_ro_progress_{dataset_name}.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")

    return results


# ---------------------------------------------------------------------------
# Summary table figure (required by assignment)
# ---------------------------------------------------------------------------

def save_summary_table(adult_results, wine_results, seeds):
    """
    Generate the required summary table figure:
    Method | Best Val Loss | Test Metric | #Func Evals
    Covers all Part 1 RO algorithms on both datasets.
    Saved as a matplotlib table figure so it appears in the figures directory.
    """
    rows = []
    headers = ['Dataset', 'Algorithm', 'Median Test F1', 'IQR F1',
               'Budget (func evals)']

    for dname, res in [('Adult', adult_results), ('Wine', wine_results)]:
        for algo in RO_CONFIGS:
            med, iqr = median_iqr(res[algo]['test_f1'])
            rows.append([dname, algo, f'{med:.4f}', f'{iqr:.4f}',
                         str(RO_BUDGET)])

    fig, ax = plt.subplots(figsize=(9, len(rows) * 0.55 + 1.2))
    ax.axis('off')
    tbl = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc='center',
        loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(headers))))

    # Shade header row
    for j in range(len(headers)):
        tbl[0, j].set_facecolor('#4472C4')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    # Alternate row shading
    for i in range(1, len(rows) + 1):
        color = '#D9E1F2' if i % 2 == 0 else 'white'
        for j in range(len(headers)):
            tbl[i, j].set_facecolor(color)

    ax.set_title('Part 1 Summary — RO algorithms on Adult and Wine\n'
                 '(median ± IQR over seeds, budget matched)',
                 fontsize=10, pad=12)
    fig.tight_layout()
    fname = FIGURES_DIR / 'part1_summary_table.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_ro_experiments(seeds=(42, 43, 44)):
    print("\n=== Part 1: Randomized Optimization (Adult + Wine) ===")

    # Adult — binary classification
    print("\n-- Adult --")
    adult_loaders = load_adult()
    adult_results = run_dataset('adult', adult_loaders,
                                output_dim=2, task='binary', seeds=seeds)

    # Wine — multiclass classification
    print("\n-- Wine --")
    wine_loaders  = load_wine()
    # load_wine returns (train_loader, val_loader, test_loader, preprocessor, n_classes)
    n_classes     = wine_loaders[4]
    wine_results  = run_dataset('wine', wine_loaders,
                                output_dim=n_classes, task='multiclass', seeds=seeds)

    # Summary table covering both datasets
    save_summary_table(adult_results, wine_results, seeds)

    # Persist raw results to JSON for use in Part 4
    out = {
        'adult': {k: {'test_f1': v['test_f1']} for k, v in adult_results.items()},
        'wine':  {k: {'test_f1': v['test_f1']} for k, v in wine_results.items()},
    }
    with open(OUTPUT_DIR / 'part1_ro_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nPart 1 done.")


if __name__ == '__main__':
    run_ro_experiments()