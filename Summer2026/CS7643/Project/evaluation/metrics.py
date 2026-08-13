"""classification metrics for the anomaly (positive) class.

with ~3% positives, accuracy is a vanity metric (predict-all-normal scores 97%).
the real scoreboard is precision / recall / F1 / ROC-AUC / PR-AUC on the anomaly
class, so those are all reported and PR-AUC is called out explicitly.
"""
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> Dict:
    """all headline metrics from ground truth + predicted probabilities."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        # positive class = anomaly (label 1); zero_division guards degenerate preds
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def format_table3(m: Dict) -> str:
    """render metrics as a paper-ready Table 3 block (anomaly = positive class)."""
    c = m["confusion"]
    lines = [
        "Table 3: Final test-set metrics (anomaly = positive class)",
        f"  samples            {m['n']} ({m['n_pos']} anomalies, "
        f"{100 * m['n_pos'] / m['n']:.2f}%)",
        f"  threshold          {m['threshold']:.2f}",
        f"  accuracy           {m['accuracy']:.4f}",
        f"  precision          {m['precision']:.4f}",
        f"  recall             {m['recall']:.4f}",
        f"  F1                 {m['f1']:.4f}",
        f"  ROC-AUC            {m['roc_auc']:.4f}",
        f"  PR-AUC             {m['pr_auc']:.4f}  (report this: 3% positives)",
        f"  confusion (tn,fp,fn,tp)  {c['tn']}, {c['fp']}, {c['fn']}, {c['tp']}",
    ]
    return "\n".join(lines)
