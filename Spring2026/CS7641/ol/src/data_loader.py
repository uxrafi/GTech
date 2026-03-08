"""
Data loading and preprocessing module.

What this does:
- Loads the Adult and Wine datasets from CSV files
- Sets up preprocessing piplines for each dataset (scaling, encoding, imputation)
- Handles train/val/test spliting with stratification (60/20/20)
- Makes sure preprocessing avoids leakage - transformers only fit on training data
- Converts data into PyTorch tensors and DataLoaders for use in the MLP backbone

The preprocessing is tailored to each dataset since Adult has mixed types
(numeric + categorical) while Wine is all numeric features.

--------------------------

Assignment Requirements Covered:
- Data loading and preprocessing workflow (deterministc)
- Leakage controls: fit only on training data, seperate pipelines per dataset
- Train/val/test spliting with stratification to maintain class proportions
- Feature scaling: StandardScaler for both datasets
- One-hot encoding for Adult categorical featurs
- Missing value handling: median imputation (numeric), 'missing' catagory (categorical)
- Wine label remapping: fixes the SL pipeline bug where quality labels (3-9) were not
  remapped to contiguous 0-indexed integers as required by CrossEntropyLoss
- Supports "Methodology & reproducability" via deterministic preprocessing with fixed seeds
"""


import pandas as pd  # dataframe handling
import numpy as np  # array ops
import torch  # pytorch core
from torch.utils.data import DataLoader, TensorDataset  # batching and dataset wrapping
from sklearn.model_selection import train_test_split  # train/val/test splits
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder  # scaling and encoding
from sklearn.compose import ColumnTransformer  # apply different transforms to different columns
from sklearn.impute import SimpleImputer  # fill missing values
from sklearn.pipeline import Pipeline  # chain preprocessing steps
from paths import DATA_DIR, RANDOM_STATE  # data location and fixed seed


# Build the preprocessing pipeline for the Adult dataset.
# Adult has a mix of numeric and categorical features, so we handle them seperately.
# Numeric: fill missing with median, then standardize to mean=0, std=1
# Categorical: mark missing as its own category, then one-hot encode
# This produces ~103 features after encoding, which becomes the MLP input dimention.
def get_adult_preprocessor():
    # Split columns by type — numeric features get scaled, categoricals get encoded
    num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    cat_cols = ['workclass', 'education', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'native-country']

    # For numbers: impute missing with median then standardize
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),   # missing values get the median
        ('scaler', StandardScaler())                      # standardize to mean=0, std=1
    ])

    # For catagories: impute missing as its own category then one-hot encode
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # missing becomes its own category
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # convert to dummy varibles
    ])

    # Combine both transformers — applys the right one to the right columns
    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])
    return preprocessor


# Build the preprocessing pipeline for the Wine dataset.
# Wine only has numeric features and no missing data, so we just standardize.
# Much simpler than Adult — all 12 features are continous physicochemical measurements.
def get_wine_preprocessor():
    return StandardScaler()


# Load and preprocess the Adult Income dataset.
# Returns PyTorch DataLoaders for train/val/test splits and the fitted preprocessor.
#
# Split strategy: 60/20/20 train/val/test using stratified sampling to preserve
# the ~24% positive class rate (income >50K) accross all splits.
#
# Labels: binary — >50K mapped to 1, <=50K mapped to 0
# Output dim: 2 (for CrossEntropyLoss compatibility)
# Input dim: ~103 after one-hot encoding categorical features
def load_adult():
    df = pd.read_csv(DATA_DIR / 'adult.csv')
    X = df.drop('class', axis=1)                                          # everything except the target
    y = df['class'].str.strip().apply(lambda x: 1 if x == '>50K' else 0) # binary label: 1 if high income, 0 otherwize

    # First split off 40% as temp (will become val + test), keep 60% as train
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
    )
    # Split temp 50/50 into val and test (each becomes 20% of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )

    # Fit preprocessor on train only — no leakage into val or test
    preprocessor = get_adult_preprocessor()
    X_train = preprocessor.fit_transform(X_train)  # fit + transform on train
    X_val   = preprocessor.transform(X_val)         # transform only — no fitting
    X_test  = preprocessor.transform(X_test)        # transform only — no fitting

    # Convert to PyTorch tensors — float32 for features, long for labels (CrossEntropyLoss requirment)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.long)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val.values,   dtype=torch.long)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test.values,  dtype=torch.long)

    # Wrap in DataLoaders — train shuffled for SGD/Adam, val/test not shuffled for reproducibility
    # Batch size 64 for training matches the OL report compute budget (~10,980 gradient updates over 30 epochs)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64,  shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=256, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=256, shuffle=False)

    return train_loader, val_loader, test_loader, preprocessor


# Load and preprocess the Wine Quality dataset.
# Returns PyTorch DataLoaders for train/val/test splits, the fitted preprocessor,
# and n_classes (number of unique quality levels after remaping).
#
# Key fix from SL report: quality labels (3-9) are remapped to contiguous 0-indexed
# integers (0-6) via LabelEncoder. Without this fix, CrossEntropyLoss fails because
# it expects labels in range [0, n_classes-1]. The SL report had output dim=10
# instead of 7, causing the label-indexing bug that inflated Wine F1 to 0.608.
# The corrected baseline is Macro-F1 = 0.512.
#
# Input dim: 12 (all continous physicochemical features)
# Output dim: n_classes (typically 7 after remapping quality scores 3-9 to 0-6)
def load_wine():
    df = pd.read_csv(DATA_DIR / 'wine.csv')
    X = df.drop('quality', axis=1)   # all features including wine type indicator
    y_raw = df['quality'].values      # raw quality scores ranging from 3 to 9

    # Remap quality labels to 0-indexed contiguous integers so CrossEntropyLoss works corectly.
    # quality 3->0, 4->1, 5->2, 6->3, 7->4, 8->5, 9->6
    # This is a bug fix from the SL pipeline, not a design change.
    le = LabelEncoder()
    y = le.fit_transform(y_raw)       # shape (N,), dtype int64
    n_classes = len(le.classes_)      # used to set MLP output dimention correctly

    # Same 60/20/20 split strategy as Adult, stratified to preserve class distribution
    X_train, X_temp, y_train, y_temp = train_test_split(
        X.values, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )

    # Fit preprocessor on train only — no leakage into val or test
    preprocessor = get_wine_preprocessor()
    X_train = preprocessor.fit_transform(X_train)  # fit + transform on train
    X_val   = preprocessor.transform(X_val)         # transform only — no fitting
    X_test  = preprocessor.transform(X_test)        # transform only — no fitting

    # Convert to PyTorch tensors — float32 for features, long for labels
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long)

    # Wrap in DataLoaders — same batch sizes as Adult for consistancy across experiments
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64,  shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=256, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=256, shuffle=False)

    return train_loader, val_loader, test_loader, preprocessor, n_classes