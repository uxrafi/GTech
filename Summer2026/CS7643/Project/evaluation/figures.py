"""paper figures 2 and 3, plus a PR curve.

Fig 2 = train/val loss vs epoch (overfitting story).
Fig 3 = ROC (AUC annotated) + confusion matrix, side by side.
all consume the agreed contract: results/history.json for loss, and
y_true.npy / y_score.npy (test probabilities) for ROC / PR / confusion.
"""
from typing import Dict, List, Union

import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from plot_style import PALETTE, apply_style, save_figure
import matplotlib.pyplot as plt

# expected history.json schema (Umar emits this per epoch):
#   {"epoch":[1,..], "train_loss":[...], "val_loss":[...],
#    optional: "train_acc"/"val_acc"/other metric arrays}
# a list-of-dicts ([{"epoch":1,"train_loss":..}, ...]) is also accepted.
HistoryLike = Union[Dict[str, List[float]], List[Dict[str, float]]]


def normalize_history(history: HistoryLike) -> Dict[str, List[float]]:
    """accept dict-of-lists or list-of-dicts -> dict-of-lists."""
    if isinstance(history, list):
        keys = history[0].keys()
        history = {k: [row[k] for row in history] for k in keys}
    out = dict(history)
    if "epoch" not in out:
        n = len(out["train_loss"])
        out["epoch"] = list(range(1, n + 1))
    return out


def plot_loss_curves(history: HistoryLike, out_dir: str, name: str = "fig2_loss"):
    """Fig 2: train vs val loss per epoch."""
    apply_style()
    h = normalize_history(history)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(h["epoch"], h["train_loss"], color=PALETTE["blue"], label="train")
    ax.plot(h["epoch"], h["val_loss"], color=PALETTE["orange"], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("Training and validation loss")
    ax.legend(frameon=False)
    return save_figure(fig, out_dir, name)


def plot_roc_and_confusion(
    y_true, y_score, out_dir: str, threshold: float = 0.5, name: str = "fig3_roc_cm"
):
    """Fig 3: ROC (AUC annotated) and confusion matrix, side by side."""
    apply_style()
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, (ax_roc, ax_cm) = plt.subplots(1, 2, figsize=(8, 3.4))

    ax_roc.plot(fpr, tpr, color=PALETTE["blue"], label=f"ROC (AUC = {roc_auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], color=PALETTE["black"], linestyle="--", linewidth=1)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC curve")
    ax_roc.legend(frameon=False, loc="lower right")

    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["normal", "anomaly"])
    disp.plot(ax=ax_cm, cmap="cividis", colorbar=False, values_format="d")
    ax_cm.set_title(f"Confusion matrix (thr={threshold:.2f})")
    ax_cm.grid(False)

    return save_figure(fig, out_dir, name)


def plot_pr_curve(y_true, y_score, out_dir: str, name: str = "fig_pr"):
    """PR curve: more honest than ROC at 3% positives."""
    apply_style()
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(recall, precision, color=PALETTE["green"], label=f"PR-AUC = {ap:.3f}")
    baseline = y_true.mean()
    ax.axhline(baseline, color=PALETTE["black"], linestyle="--", linewidth=1,
               label=f"baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend(frameon=False, loc="upper right")
    return save_figure(fig, out_dir, name)
