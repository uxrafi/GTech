"""
Data loading and preprocessing module.

What this does:

- Loads the Adult and Wine datasets from CSV files
- Sets up preprocesing piplines for each dataset (scaling, encoding, imputation)
- Handles train/test splitting with stratification
- Makes sure preprocessing avoids leakage - transformers only fit on training data later

The preprocessing is tailored to each dataset since Adult has mixed types (numeric + categorical) while Wine is all numeric features.
"""

##########################

"""
Assignment Requirements Covered:

- Data loading and preprocessing workflow (deterministc)
- Leakage controls: fit only on training data, seperate pipelines per dataset
- Train/test spliting with  stratification to maintain  class proportions
- Feature scaling: StandardScaler for both datasets
- One-hot encoding for Adult categorical featurs
- Missing value handling: median imputation (numeric), 'missing' catagory (categorical)
- Supports "Methodology & reproducability"  via deterministic preprocessing
"""

import pandas as pd  # reading CSVs
from sklearn.model_selection import train_test_split  # spliting into train/test
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # scaling numbers and encoding catagories
from sklearn.compose import ColumnTransformer  # lets us preprocess different column types differently
from sklearn.impute import SimpleImputer  # fills in missing values
from sklearn.pipeline import Pipeline  # chains multiple steps togeather
from src.paths import DATA_DIR, RANDOM_STATE  # where data lives and our fixed seed


# Load Adult dataset and return features and target
def load_adult():
    df = pd.read_csv(DATA_DIR / 'adult.csv')
    X = df.drop('class', axis=1)  # everything except the target
    y = df['class'].apply(lambda x: 1 if x == '>50K' else 0)  # 1 if high income, 0 otherwize
    return X, y


# Load Wine dataset and return features and target
def load_wine():
    df = pd.read_csv(DATA_DIR / 'wine.csv')
    # The 'type' column (red=0, white=1) is allready in there
    X = df.drop('quality', axis=1)  # all features including wine type
    y = df['quality'].values  # quality scores range 3-9
    return X, y



# Build the preprcessing pipeline for Adult dataset.
# Adult has a mix of numeric and categorical features, so we handle them seperately.
# Numeric: fill missing with median, then standardize
# Categorical: mark missing as its own catagory, then one-hot encode
def get_adult_preprocessor():
    # Split columns by type
    num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    cat_cols = ['workclass', 'education', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'native-country']
    
    # For numbers: impute then scale
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # missing values get the median
        ('scaler', StandardScaler())  # standardize to mean=0, std=1
    ])
    
    # For catagories: impute then encode
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # missing becomes its own category
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # convert to dummy varibles
    ])
    
    # Stick both together - this applys the right transformer to the right columns
    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])
    return preprocessor


# Build the preprocessing pipeline for Wine dataset.
# Wine only has numeric features and no missing data, so we just need to standardize.
# Much simpler than Adult.
def get_wine_preprocessor():
    return StandardScaler()



# Split into 80/20 train/test, with stratification to keep class proportions.
# Stratification is important here since both datasets are imbalanced - 
# Adult has 75/25 income split, Wine quality is clusterd around 5-6.
def split_data(X, y, stratify=True):
    strat = y if stratify else None  # stratify on target if requested
    return train_test_split(X, y, test_size=0.2, stratify=strat, random_state=RANDOM_STATE)