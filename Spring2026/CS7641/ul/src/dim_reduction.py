"""
Dimensionality reducton module: PCA, ICA, Random Projecton, and UMAP.

What this does:

- fit_pca(): fits PCA, plots cumulative explained varince curve, returns model + data
- fit_ica(): fits FastICA with PCA whitning, plots kurtosis per component
- fit_rp(): fits Gaussian Random Projecton, sweeps output dims and plots recon error
- fit_umap(): fits UMAP for 2D visualizaton only (not used as fetures downstream)
- fit_all_dr(): conveniece wrapper that runs all three linear methods for one dataest

Component count selecton is label-free throughout:
  PCA  – elbow in cumulative explained varince (we pick where curve flattens)
  ICA  – absolute excess kurtosis per component (higher = more non-Gausian = better)
  RP   – reconstruction error via pseudoinverse swept across output dimensions
"""


import warnings
warnings.filterwarnings("ignore")  # suppress convergence warnings during ICA

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kurtosis as sp_kurtosis

from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection

from src.paths import FIGURES_DIR, RANDOM_STATE

try:
    import umap as umap_lib
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def fit_pca(X, n_components, name=""):
    """Fit PCA, plot cumulative variance, return model and transformed data."""
    pca_full = PCA(random_state=RANDOM_STATE).fit(X)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(cumvar) + 1), cumvar, marker="o", ms=4)
    ax.axhline(0.90, ls="--", color="gray", label="90% variance")
    ax.axhline(0.95, ls=":", color="red", label="95% variance")
    ax.axvline(n_components, ls="--", color="steelblue", label=f"Chosen n={n_components}")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title(f"PCA Explained Variance – {name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"pca_variance_{name.replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE).fit(X)
    Xt = pca.transform(X)
    recon_mse = np.mean((X - pca.inverse_transform(Xt)) ** 2)

    print(f"  [{name} / PCA] n={n_components}  "
          f"var_explained={cumvar[n_components - 1]:.3f}  "
          f"recon_MSE={recon_mse:.4f}")
    return pca, Xt


def fit_ica(X, n_components, name=""):
    """Fit FastICA, compute kurtosis per component, plot results."""
    ica = FastICA(
        n_components=n_components,
        whiten="unit-variance",
        fun="logcosh",
        max_iter=500,
        random_state=RANDOM_STATE
    )
    Xt = ica.fit_transform(X)

    kurt_vals = [sp_kurtosis(Xt[:, i], fisher=True)
                 for i in range(Xt.shape[1])]

    print(f"  [{name} / ICA] n={n_components}  "
          f"kurtosis: {[f'{v:.2f}' for v in kurt_vals]}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, n_components + 1), kurt_vals, color="steelblue")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("ICA Component")
    ax.set_ylabel("Excess Kurtosis (Fisher)")
    ax.set_title(f"ICA Component Kurtosis – {name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"ica_kurtosis_{name.replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    return ica, Xt


def fit_rp(X, n_components, name=""):
    """Fit Gaussian Random Projection, sweep dims to show reconstruction error."""
    sweep_dims = list(range(2, min(X.shape[1] + 1, 22), 2))
    mse_means, mse_stds = [], []

    for d in sweep_dims:
        seed_mses = []
        for s in range(5):
            rp_s = GaussianRandomProjection(
                n_components=d, random_state=RANDOM_STATE + s).fit(X)
            Xt_s = rp_s.transform(X)
            R = rp_s.components_
            R_pinv = np.linalg.pinv(R)
            Xr = Xt_s @ R_pinv.T
            seed_mses.append(np.mean((X - Xr) ** 2))
        mse_means.append(np.mean(seed_mses))
        mse_stds.append(np.std(seed_mses))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(sweep_dims, mse_means, yerr=mse_stds,
                marker="o", capsize=4, label="Recon MSE ± std (5 seeds)")
    ax.axvline(n_components, ls="--", color="red", label=f"Chosen n={n_components}")
    ax.set_xlabel("Output Dimension")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title(f"RP Reconstruction Error – {name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"rp_recon_{name.replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    rp = GaussianRandomProjection(
        n_components=n_components, random_state=RANDOM_STATE).fit(X)
    Xt = rp.transform(X)
    R = rp.components_
    R_pinv = np.linalg.pinv(R)
    recon_mse = np.mean((X - Xt @ R_pinv.T) ** 2)

    chosen_std = (mse_stds[sweep_dims.index(n_components)]
                  if n_components in sweep_dims else float("nan"))

    print(f"  [{name} / RP ] n={n_components}  "
          f"recon_MSE={recon_mse:.4f}  seed_std={chosen_std:.4f}")
    return rp, Xt


def fit_all_dr(X, n_components, name=""):
    """Convenience wrapper: run PCA, ICA, RP for one dataset."""
    models, data = {}, {}

    pca, Xt_pca = fit_pca(X, n_components, name)
    models["pca"] = pca
    data["pca"] = Xt_pca

    ica, Xt_ica = fit_ica(X, n_components, name)
    models["ica"] = ica
    data["ica"] = Xt_ica

    rp, Xt_rp = fit_rp(X, n_components, name)
    models["rp"] = rp
    data["rp"] = Xt_rp

    return models, data


def fit_umap(X, y, name=""):
    """Fit UMAP for 2D visualization (extra credit)."""
    if not UMAP_AVAILABLE:
        print(f"  [SKIP] UMAP not available for {name}. pip install umap-learn")
        return

    reducer = umap_lib.UMAP(
        n_neighbors=15, min_dist=0.1,
        n_components=2, random_state=RANDOM_STATE
    )
    emb = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=y,
                    cmap="tab10", s=3, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="True Label")
    ax.set_title(f"UMAP 2D Embedding – {name}")
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    fig.tight_layout()
    fname = FIGURES_DIR / f"umap_{name.lower().replace(' ', '_')}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{name} / UMAP] saved {fname.name}")