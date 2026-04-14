"""
Step 1: clustering on raw feature spaces.

Sweeps k for KMeans and GMM on both datasets, picks final k,
fits and returns the four models.

Wine: going with k=4 not k=2 -- k=2 just splits red/white which is
boring and basically already captured by the color feature. k=4 gets
at quality tiers which is actually interesting.

Adult: silhouette is pretty bad across the board, sparse binary features
make euclidean distance kind of meaningless. k=2 is the only sane pick.
"""

from src.clustering import sweep_k, fit_final_models, eval_clustering, fmt_metrics


def run_step1(X_wine, y_wine, X_adult, y_adult):
    print("\n" + "=" * 60)
    print("STEP 1: Clustering on raw data")
    print("=" * 60)

    df_wine  = sweep_k(X_wine,  y_wine,  name="Wine Raw")
    df_adult = sweep_k(X_adult, y_adult, name="Adult Raw")

    print("\n[Wine]  best silhouette per algo:")
    best_w = df_wine.loc[df_wine.groupby("algo")["silhouette"].idxmax()]
    print(best_w[["algo", "k", "silhouette", "ch", "db", "ari"]].to_string(index=False))

    print("\n[Adult] best silhouette per algo:")
    best_a = df_adult.loc[df_adult.groupby("algo")["silhouette"].idxmax()]
    print(best_a[["algo", "k", "silhouette", "ch", "db", "ari"]].to_string(index=False))

    km_wine,  gmm_wine  = fit_final_models(X_wine,  k=4)
    km_adult, gmm_adult = fit_final_models(X_adult, k=2)

    # quick sanity check on final models
    print(f"\n[Wine]  KMeans k=4  | " + fmt_metrics(eval_clustering(X_wine, km_wine.labels_, y_wine)))
    print(f"[Wine]  GMM    k=4  | " + fmt_metrics(eval_clustering(X_wine, gmm_wine.predict(X_wine), y_wine)))
    print(f"[Adult] KMeans k=2  | " + fmt_metrics(eval_clustering(X_adult, km_adult.labels_, y_adult)))
    print(f"[Adult] GMM    k=2  | " + fmt_metrics(eval_clustering(X_adult, gmm_adult.predict(X_adult), y_adult)))

    return km_wine, gmm_wine, km_adult, gmm_adult