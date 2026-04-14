"""
Clustering helpers used across Step 1 and Step 3.

What this does:

- Provides eval_clustering() which computes all four metrcs at once
- Provides sweep_k() which sweeps k for both KMeans and GMM and saves plots
- Provides fit_final_models() which fits the chosen k after the sweep is done
- Keeps all clustring logic in one place so Step1 and Step3 dont duplicte code

The three label-free metrcs we use for selecton are silhouete score, Calinski-
Harabsz index, and Davies-Bouldin index. ARI is computed but only used post-hoc
for interpretaton in the reprot - we never use it to pick k becuse that would
leak label informaton into the unsupervsd selecton process.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score
)

from src.paths import FIGURES_DIR, RANDOM_STATE


def fmt_metrics(m):
    """Quick print helper for metric dict."""
    return (f"sil={m['silhouette']:.3f}  ch={m['ch']:.1f}  "
            f"db={m['db']:.3f}  ari={m['ari']:.3f}")


def eval_clustering(X, labels, y_true=None):
    """Compute silhouette, CH, DB, and optional ARI."""
    return {
        "silhouette": silhouette_score(X, labels),
        "ch": calinski_harabasz_score(X, labels),
        "db": davies_bouldin_score(X, labels),
        "ari": adjusted_rand_score(y_true, labels) if y_true is not None else np.nan,
    }


def sweep_k(X, y_true, k_range=range(2, 14), name=""):
    """
    Sweep k for KMeans and GMM, save metric plots, return DataFrame.
    Uses diagonal covariance for GMM (full was unstable for Wine at k=4).
    """
    X64 = X.astype("float64")
    rows = []

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X64)
        gmm = GaussianMixture(
            n_components=k, n_init=5, covariance_type="diag",
            reg_covar=1e-3, random_state=RANDOM_STATE
        ).fit(X64)

        for algo, labels in [("KMeans", km.labels_),
                              ("GMM",   gmm.predict(X64))]:
            m = eval_clustering(X64, labels, y_true)
            m["k"] = k
            m["algo"] = algo
            rows.append(m)

    df = pd.DataFrame(rows)

    # 3-panel plot: silhouette, CH, DB vs k
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, ylabel in zip(
            axes,
            ["silhouette", "ch", "db"],
            ["Silhouette Score", "Calinski-Harabasz Index", "Davies-Bouldin Index"]):
        for algo, grp in df.groupby("algo"):
            ax.plot(grp["k"], grp[metric], marker="o", label=algo)
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}\n{name}")
        ax.legend()
    fig.suptitle(f"Clustering Metric Sweep – {name}", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"cluster_sweep_{name.replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    return df


def fit_final_models(X, k):
    """Fit KMeans and GMM at chosen k, return both models."""
    X64 = X.astype("float64")
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X64)
    gmm = GaussianMixture(
        n_components=k, n_init=5, covariance_type="diag",
        reg_covar=1e-3, random_state=RANDOM_STATE
    ).fit(X64)
    return km, gmm