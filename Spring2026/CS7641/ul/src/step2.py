"""
Step 2: dimensionality reduction on raw features.

Runs PCA, ICA, RP on both datasets and UMAP just for visualization.
Wine -> 4 components, Adult -> 20.

Component counts picked by label-free criteria (variance elbow, kurtosis
dropoff, RP reconstruction error). Using the same n for all three methods
per dataset so the step 3 comparisons are actually fair.
"""

from src.dim_reduction import fit_all_dr, fit_umap


def run_step2(X_wine, y_wine, X_adult, y_adult):
    print("\n" + "=" * 60)
    print("STEP 2: Dimensionality Reduction")
    print("=" * 60)

    wine_models,  wine_data  = fit_all_dr(X_wine,  n_components=4,  name="Wine")
    adult_models, adult_data = fit_all_dr(X_adult, n_components=20, name="Adult")

    # umap is just for the visualization plots, not used as features anywhere
    fit_umap(X_wine,  y_wine,  name="Wine")
    fit_umap(X_adult, y_adult, name="Adult")

    dr_models, dr_data = {}, {}
    for method in ["pca", "ica", "rp"]:
        dr_models[("wine",  method)] = wine_models[method]
        dr_models[("adult", method)] = adult_models[method]
        dr_data[("wine",    method)] = wine_data[method]
        dr_data[("adult",   method)] = adult_data[method]

    return dr_models, dr_data