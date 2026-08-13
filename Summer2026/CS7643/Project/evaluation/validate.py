"""independent validation of the trained model + paper figure generation.

this is the referee step: load the test-set ground truth and the model's predicted
probabilities, recompute every metric with scikit-learn, and render Figures 2-3.
running it against Umar's outputs catches metric bugs or train/test contamination
before they reach the paper.

usage (once Umar delivers his Week-2 artifacts):
    python evaluation/validate.py \
        --history results/history.json \
        --y-true results/y_true.npy --y-score results/y_score.npy \
        --out results/plots
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from figures import plot_loss_curves, plot_pr_curve, plot_roc_and_confusion
from metrics import compute_metrics, format_table3


def infer_scores(model, X, lengths, batch_size: int = 1024):
    """batched inference -> per-sequence anomaly probability.

    for use when Umar's model is importable: pass a loaded torch model in eval
    mode. kept dependency-light (torch imported lazily) so the metrics/figure path
    works from y_score.npy alone without torch installed.
    """
    import torch

    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[i : i + batch_size], dtype=torch.long)
            lb = torch.as_tensor(lengths[i : i + batch_size], dtype=torch.long)
            logits = model(xb, lb).squeeze(-1)
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def main():
    ap = argparse.ArgumentParser(description="independent model validation + figures")
    ap.add_argument("--history", help="results/history.json (per-epoch loss/metrics)")
    ap.add_argument("--y-true", required=True, help="test-set labels .npy")
    ap.add_argument("--y-score", required=True, help="test-set probabilities .npy")
    ap.add_argument("--out", default="results/plots", help="figure output dir")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    y_true = np.load(args.y_true)
    y_score = np.load(args.y_score)
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_score {y_score.shape}")

    metrics = compute_metrics(y_true, y_score, args.threshold)
    print(format_table3(metrics))

    made = []
    if args.history:
        with open(args.history) as f:
            history = json.load(f)
        made.append(plot_loss_curves(history, args.out))
    made.append(plot_roc_and_confusion(y_true, y_score, args.out, args.threshold))
    made.append(plot_pr_curve(y_true, y_score, args.out))

    print("\nfigures written:")
    for p in made:
        print(f"  {p}")

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
