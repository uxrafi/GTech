"""
Quick data inspection script.

What this does:
- It loads Adult and Wine datasets and prints basic info
- Shows dataset shape (rows x columns)
- Lists all column names
- Counts missing values per column
- Shows target variable distributions

Usefull for a quick sanity check before running full experiments.
"""

##########################

"""
Assignment Requirements Covered:

- Partial EDA: basic data inspection (shape, colums, missing values, distributions)
- NOT a full exploratory data analisys - deeper analysis needed in report
- Helps verify data loaded correctly and looks reasonable before experiments
- Target distribution check for understanding class imbalance
"""

import pandas as pd  # for reading and inspecting CSVs
from paths import DATA_DIR  # where our data files are stored


# Print basic info about the Adult dataset
# Shows shape, columns, missing values, and class distrbution
def inspect_adult():
    df = pd.read_csv(DATA_DIR / 'adult.csv')
    print("Adult dataset:")
    print(f"Shape: {df.shape}")  # rows and colums
    print(f"Columns: {df.columns.tolist()}")
    print("Missing values:\n", df.isnull().sum())  # count NaNs per column
    print("Class distribution:\n", df['class'].value_counts())  # income split


# Print basic info about the Wine dataset
# Same as above but for wine quality ratigns
def inspect_wine():
    df = pd.read_csv(DATA_DIR / 'wine.csv')
    print("Wine dataset:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("Missing values:\n", df.isnull().sum())  # should be zero for wine
    print("Class distribution:\n", df['quality'].value_counts().sort_index())  # quality 3-9



# Run both inspections when script is executed direclty
if __name__ == '__main__':
    inspect_adult()
    print("\n" + "="*50 + "\n")  # seperator between datasets
    inspect_wine()