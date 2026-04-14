"""
Step 3: re-run clustering in each reduced feature space.

Key change from previous version: k is now re-tuned per reduced space
using the same label-free sweep procedure as Step 1. The TA confirmed
that optimal k may differ in the transformed space and must be re-selected.

Wine and Adult each get their own k sweep per DR method, so we may end up
with different k values across PCA/ICA/RP spaces.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from src.clustering import sweep_k, eval_clustering, fmt_metrics
from src.paths import FIGURES_DIR, RESULTS_DIR, RANDOM_STATE


def _select_k(df_sweep, algo):
    """
    Pick best k from sweep results using silhouette score (label-free).
    Returns the k with the highest silhouette for the given algo.
    """
    sub = df_sweep[df_sweep["algo"] == algo]
    best_row = sub.loc[sub["silhouette"].idxmax()]
    return int(best_row["k"])


def run_step3(dr_data, y_wine, y_adult):
    print("\n" + "=" * 60)
    print("STEP 3: Clustering in reduced spaces (k re-tuned per space)")
    print("=" * 60)

    ys = {"wine": y_wine, "adult": y_adult}
    rows = []

    for (ds, method), Xt in dr_data.items():
        y = ys[ds]

        # Re-run k sweep in this reduced space — same procedure as Step 1
        print(f"\n  [{ds.upper()} / {method.upper()}] sweeping k...")
        df_sweep = sweep_k(
            Xt, y,
            k_range=range(2, 14),
            name=f"{ds.title()} {method.upper()}"
        )

        # Select best k per algorithm using silhouette (label-free)
        k_km  = _select_k(df_sweep, "KMeans")
        k_gmm = _select_k(df_sweep, "GMM")
        print(f"    Selected k: KMeans={k_km}, GMM={k_gmm}")

        # Fit final models at selected k
        km = KMeans(
            n_clusters=k_km, n_init=20, random_state=RANDOM_STATE
        ).fit(Xt)

        gmm = GaussianMixture(
            n_components=k_gmm, n_init=5, covariance_type="diag",
            reg_covar=1e-3, random_state=RANDOM_STATE
        ).fit(Xt.astype("float64"))

        for algo, labels, k_sel in [
            ("KMeans", km.labels_,          k_km),
            ("GMM",    gmm.predict(Xt),     k_gmm),
        ]:
            m = eval_clustering(Xt, labels, y)
            m.update({
                "dataset": ds,
                "dr":      method.upper(),
                "algo":    algo,
                "k":       k_sel,
            })
            rows.append(m)
            print(f"  [{ds.upper():5s} / {method.upper():3s} / {algo:6s} k={k_sel}] "
                  + fmt_metrics(m))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "step3_results.csv", index=False)

    # Plot silhouette scores per reduced space
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, ds in zip(axes, ["wine", "adult"]):
        sub = df[df["dataset"] == ds].copy()
        sub["label"] = sub["dr"] + "\n" + sub["algo"] + "\n(k=" + sub["k"].astype(str) + ")"
        colors = ["steelblue" if a == "KMeans" else "coral" for a in sub["algo"]]
        ax.bar(sub["label"], sub["silhouette"], color=colors)
        ax.set_title(f"Silhouette in Reduced Spaces – {ds.title()}\n(k re-tuned per space)")
        ax.set_ylabel("Silhouette Score")
        ax.set_xlabel("DR Method / Algorithm")
        ax.tick_params(axis="x", labelsize=7)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "step3_silhouette_comparison.png",
        dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    return df