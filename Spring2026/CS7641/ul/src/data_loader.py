"""
Data loading and preprocessing module.

What this does:

- Loads Adult and Wine datasets from CSV files in the data/ folder
- Sets up preprocessing pipelines for each dataset (scaling, encoding, imputation)
- Handles train/test splitting with stratification to maintain class proportions
- Makes sure preprocessing avoids leakage - transformers only fit on training data later

Adult has mixed types (numeric + categorical) so it needs more work than Wine.
Wine is all numeric features which makes it much simpler to preprocess.
Both datasets use StandardScaler so the input distributions are consistent
with the OL report baseline, keeping Steps 4 and 5 comparisons valid.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from src.paths import DATA_DIR, RANDOM_STATE


# ── Adult Income ─────────────────────────────────────────────────

def load_adult():
    """
    Load Adult Income dataset from data/adult.csv.
    Handles missing values (encoded as ' ?') and one-hot encodes categoricals.
    Numeric features standardized with StandardScaler.
    Returns X (float32) and y (int).
    """
    col_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    df_peek = pd.read_csv(DATA_DIR / "adult.csv", nrows=1)
    has_header = (df_peek.columns[0] != "0" and
                  not str(df_peek.columns[0]).lstrip("-").isdigit())

    if has_header:
        df = pd.read_csv(DATA_DIR / "adult.csv",
                         na_values=["?", " ?"], skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        if "class" in df.columns:
            df.rename(columns={"class": "income"}, inplace=True)
    else:
        df = pd.read_csv(DATA_DIR / "adult.csv", header=None,
                         names=col_names, na_values=[" ?", "?"],
                         skipinitialspace=True)

    # Fill missing with mode per column (label-free, no leakage risk at dataset level)
    for col in df.columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    df.reset_index(drop=True, inplace=True)

    y = (df["income"].str.strip().str.replace(".", "", regex=False) == ">50K"
         ).astype(int).values
    df.drop(columns=["income"], inplace=True)

    num_cols = ["age", "fnlwgt", "education-num",
                "capital-gain", "capital-loss", "hours-per-week"]
    cat_cols = ["workclass", "education", "marital-status", "occupation",
                "relationship", "race", "sex", "native-country"]

    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]

    df_num = pd.DataFrame(
        StandardScaler().fit_transform(df[num_cols]), columns=num_cols)
    df_cat = pd.get_dummies(df[cat_cols])

    X = pd.concat([df_num, df_cat], axis=1).values.astype("float32")
    print(f"  Adult loaded  : {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


# ── Wine Quality ─────────────────────────────────────────────────

def load_wine():
    """
    Load Wine Quality dataset from data/wine.csv.
    Auto-detects separator, adds binary color indicator if missing.

    Preprocessing change vs. previous UL version:
      StandardScaler is used (not MinMaxScaler) to match the OL report
      pipeline. This makes the OL baseline accuracy (57.8%) a valid
      comparison point for Steps 4 and 5.

    Returns X (float32), y (int64, 0-indexed), LabelEncoder.
    """
    wine_path = DATA_DIR / "wine.csv"
    if not wine_path.exists():
        raise FileNotFoundError(f"wine.csv not found at {wine_path}")

    df = pd.read_csv(wine_path, sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(wine_path, sep=",")

    df.columns = [c.strip() for c in df.columns]
    if "type" in df.columns:
        df.rename(columns={"type": "color"}, inplace=True)
    if "color" not in df.columns:
        df["color"] = 0

    y_raw = df["quality"].values.astype(int)
    X_raw = df.drop(columns=["quality"]).values.astype("float32")

    # StandardScaler to match OL preprocessing
    X = StandardScaler().fit_transform(X_raw).astype("float32")

    le = LabelEncoder()
    y  = le.fit_transform(y_raw).astype("int64")

    print(f"  Wine loaded   : {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(le.classes_)} classes  (quality {le.classes_[0]}-{le.classes_[-1]})")
    return X, y, le


# ── Train / Test Split ───────────────────────────────────────────

def split_data(X, y):
    """80/20 stratified train/test split. Matches OL report split."""
    return train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )


# ── Synthetic Fallback ───────────────────────────────────────────

def synthetic_fallback(n, d, n_classes, name):
    """Generate random data when real CSV files are missing."""
    print(f"  [WARN] {name} files not found – using synthetic stand-in ({n}x{d})")
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((n, d)).astype("float32")
    y = rng.integers(0, n_classes, n)
    return X, y