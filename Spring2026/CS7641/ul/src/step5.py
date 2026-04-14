"""
Step 5: retrain NN on cluster-derived Wine features.

Three encodings from the step 1 models:
  - KM hard labels      : one-hot cluster assignment (4 binary cols)
  - KM centroid distances: euclidean dist to each centroid (4 continuous)
  - EM posteriors       : GMM soft responsibilities (4 continuous, sum=1)

Each tested two ways:
  augmented = original 12 + cluster features (16 total)
  replaced  = cluster features only (4 total)

Expecting augmented > replaced since clusters alone are too coarse to
carry enough info for 7-class quality prediction on their own.
Also expecting EM posteriors > centroid distances > hard labels since
each encoding throws away progressively more within-cluster structure.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.neural_net import train_nn
from src.paths import FIGURES_DIR, RESULTS_DIR, RANDOM_STATE


def make_cluster_features(X, km_model, gmm_model):
    k    = km_model.n_clusters
    X32  = X.astype("float32")
    X64  = X.astype("float64")

    km_labels = km_model.predict(X64)
    km_onehot = np.eye(k)[km_labels].astype("float32")

    # distances to each centroid
    diffs   = X32[:, None, :] - km_model.cluster_centers_.astype("float32")[None, :, :]
    km_dist = np.linalg.norm(diffs, axis=2).astype("float32")

    em_post = gmm_model.predict_proba(X64).astype("float32")

    return {
        "KM hard labels":       km_onehot,
        "KM centroid distances": km_dist,
        "EM posteriors":         em_post,
    }


def run_step5(X_wine, y_wine, km_wine, gmm_wine):
    print("\n" + "=" * 60)
    print("STEP 5: Neural network on cluster-derived Wine features")
    print("=" * 60)

    cf    = make_cluster_features(X_wine, km_wine, gmm_wine)
    n_cls = len(np.unique(y_wine))

    idx = np.arange(len(X_wine))
    tr_idx, te_idx = train_test_split(
        idx, test_size=0.2, stratify=y_wine, random_state=RANDOM_STATE)

    Xtr_raw, Xte_raw = X_wine[tr_idx], X_wine[te_idx]
    ytr, yte = y_wine[tr_idx], y_wine[te_idx]

    rows = []
    for fname, feats in cf.items():
        ftr, fte = feats[tr_idx], feats[te_idx]

        # augmented: stack cluster features on top of originals
        Xtr_aug = np.concatenate([Xtr_raw, ftr], axis=1)
        Xte_aug = np.concatenate([Xte_raw, fte], axis=1)
        acc, f1, _, _ = train_nn(Xtr_aug, ytr, Xte_aug, yte, n_cls)
        rows.append({"features": fname, "mode": "Augmented", "acc": acc, "f1": f1})
        print(f"  {fname:25s} AUG  acc={acc:.4f}  f1={f1:.4f}")

        # replaced: cluster features only, no originals
        acc, f1, _, _ = train_nn(ftr, ytr, fte, yte, n_cls)
        rows.append({"features": fname, "mode": "Replaced", "acc": acc, "f1": f1})
        print(f"  {fname:25s} REP  acc={acc:.4f}  f1={f1:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "step5_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    labels  = df["features"].unique()
    x, w    = np.arange(len(labels)), 0.35
    aug_acc = df[df["mode"] == "Augmented"]["acc"].values
    rep_acc = df[df["mode"] == "Replaced"]["acc"].values

    ax.bar(x - w/2, aug_acc, w, label="Augmented", color="steelblue")
    ax.bar(x + w/2, rep_acc, w, label="Replaced",  color="coral")
    ax.axhline(0.578, ls="--", color="gray", label="Raw baseline (0.578)")  # from OL report
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Step 5: Cluster Feature NN Comparison")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step5_cluster_features.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return df