"""logistic-regression baseline: bag-of-words event counts -> anomaly probability.

turns each block's full (untruncated) event sequence into a count vector over
the 29 HDFS templates and fits sklearn LogisticRegression(class_weight='balanced').
reuses the same train/val/test block split as the sequence models (splits.json)
so results are directly comparable, and reuses evaluation/metrics.py + figures.py
for scoring so Table 3 and Fig 3 stay one implementation across every model.

usage:
    python models/logistic_regression.py --raw data/raw --splits data/sequences/splits.json
"""
import argparse
import json
import os
import sys

import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "preprocessing"))
sys.path.insert(0, os.path.join(_ROOT, "evaluation"))

from labeler import attach_labels, load_labels  # noqa: E402
from log_parser import parse_event_traces  # noqa: E402
from sequence_creator import PAD_TOKEN, UNK_TOKEN, build_vocab  # noqa: E402
from metrics import compute_metrics, format_table3  # noqa: E402
from figures import plot_pr_curve, plot_roc_and_confusion  # noqa: E402

# selected on validation ROC-AUC (the project's headline target metric);
# val PR-AUC is logged alongside since it's the more honest read at 3% positives
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]


def event_columns(vocab):
    """ordered event tokens (excludes PAD/UNK), matching vocab id order -> E1..E29."""
    items = sorted(
        ((tok, idx) for tok, idx in vocab.items() if tok not in (PAD_TOKEN, UNK_TOKEN)),
        key=lambda kv: kv[1],
    )
    return [tok for tok, _ in items]


def build_count_matrix(traces, columns):
    """[N, len(columns)] bag-of-words event counts over full (untruncated) sequences."""
    col_idx = {tok: i for i, tok in enumerate(columns)}
    X = np.zeros((len(traces), len(columns)), dtype=np.float64)
    for i, (_, seq) in enumerate(traces):
        for tok in seq:
            j = col_idx.get(tok)
            if j is not None:
                X[i, j] += 1
    return X


def select_c(X_train, y_train, X_val, y_val, grid):
    """small grid search over C; model selection on validation ROC-AUC."""
    results = []
    best_c, best_score = None, -1.0
    for c in grid:
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=1000)
        clf.fit(X_train, y_train)
        score = clf.predict_proba(X_val)[:, 1]
        m = compute_metrics(y_val, score)
        results.append({"C": c, "val_roc_auc": m["roc_auc"], "val_pr_auc": m["pr_auc"]})
        if m["roc_auc"] > best_score:
            best_c, best_score = c, m["roc_auc"]
    return best_c, results


def write_coefficients(model, columns, X_train, y_train, path):
    """rank events by coefficient, alongside each event's train count difference
    (mean in anomaly minus mean in normal). counts are unstandardized and several
    events are correlated, so raw coefficients are an imperfect importance ranking
    on their own; the frequency column is the order-free sanity check."""
    freq_diff = X_train[y_train == 1].mean(axis=0) - X_train[y_train == 0].mean(axis=0)
    ranked = sorted(
        zip(columns, model.coef_[0], freq_diff), key=lambda kv: kv[1], reverse=True
    )
    with open(path, "w") as f:
        f.write("event  coefficient  anom_minus_normal_count  (positive coef -> anomaly)\n")
        for tok, coef, fd in ranked:
            f.write(f"{tok:<6} {coef:+.4f}  {fd:+.4f}\n")


def main():
    ap = argparse.ArgumentParser(description="logistic-regression bag-of-words baseline")
    ap.add_argument("--raw", default="data/raw", help="dir with the raw csvs")
    ap.add_argument("--splits", default="data/sequences/splits.json", help="block-id split file")
    ap.add_argument("--out", default="results/logreg", help="metrics/model output dir")
    ap.add_argument("--plots-out", default="results/plots/logreg", help="figure output dir")
    args = ap.parse_args()

    traces_path = os.path.join(args.raw, "Event_traces.csv")
    labels_path = os.path.join(args.raw, "anomaly_label.csv")
    templates_path = os.path.join(args.raw, "HDFS.log_templates.csv")

    print(f"[1/5] parsing full event sequences from {traces_path}")
    traces = parse_event_traces(traces_path)
    block_to_row = {block_id: i for i, (block_id, _) in enumerate(traces)}

    print(f"[2/5] building bag-of-words count matrix from {templates_path}")
    vocab = build_vocab(templates_path)
    columns = event_columns(vocab)
    X = build_count_matrix(traces, columns)

    print(f"[3/5] labeling from {labels_path}")
    label_map = load_labels(labels_path)

    print(f"[4/5] loading splits from {args.splits}")
    with open(args.splits) as f:
        splits = json.load(f)

    data = {}
    for name, block_ids in splits.items():
        rows = np.array([block_to_row[b] for b in block_ids], dtype=np.int64)
        data[f"X_{name}"] = X[rows]
        data[f"y_{name}"] = attach_labels(block_ids, label_map)

    print(f"[5/5] grid search over C={C_GRID}, selecting by validation ROC-AUC")
    best_c, grid_results = select_c(
        data["X_train"], data["y_train"], data["X_val"], data["y_val"], C_GRID
    )
    print("validation grid (Table 2):")
    for r in grid_results:
        print(f"  C={r['C']:<8} val_roc_auc={r['val_roc_auc']:.4f}  val_pr_auc={r['val_pr_auc']:.4f}")
    print(f"selected C={best_c}")

    # final fit on train only -- val stays held-out, mirroring how val is used
    # for model selection (not merged into training) elsewhere in the project
    print("refitting final model on train at selected C")
    final = LogisticRegression(C=best_c, class_weight="balanced", max_iter=1000)
    final.fit(data["X_train"], data["y_train"])

    y_score = final.predict_proba(data["X_test"])[:, 1]
    metrics = compute_metrics(data["y_test"], y_score)
    print()
    print(format_table3(metrics))

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "y_score.npy"), y_score)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"selected_C": best_c, "c_grid": grid_results, "test": metrics}, f, indent=2)
    dump(final, os.path.join(args.out, "model.joblib"))
    write_coefficients(final, columns, data["X_train"], data["y_train"], os.path.join(args.out, "coefficients.txt"))

    made = [
        plot_roc_and_confusion(data["y_test"], y_score, args.plots_out),
        plot_pr_curve(data["y_test"], y_score, args.plots_out),
    ]
    print("\nfigures written:")
    for p in made:
        print(f"  {p}")


if __name__ == "__main__":
    main()
