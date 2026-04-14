"""
Main pipeline for CS 7641 unsupervised learning assignment.
Run with: python main.py
Needs adult.csv and wine.csv in data/ folder.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from src.paths import RANDOM_STATE

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

from src.data_loader  import load_adult, load_wine, synthetic_fallback
from src.step1        import run_step1
from src.step2        import run_step2
from src.step3        import run_step3
from src.step4        import run_step4
from src.step5        import run_step5


def main():
    print("CS 7641 – Unsupervised Learning Pipeline")
    print("urafi3 / Umar Rafi")
    print("=" * 60)

    print("\nLoading datasets...")
    try:
        X_adult, y_adult = load_adult()
    except Exception as e:
        print(f"  adult load failed: {e}, using synthetic fallback")
        X_adult, y_adult = synthetic_fallback(5000, 108, 2, "Adult")

    try:
        X_wine, y_wine, _ = load_wine()
    except Exception as e:
        print(f"  wine load failed: {e}, using synthetic fallback")
        X_wine, y_wine = synthetic_fallback(1000, 12, 7, "Wine")

    # step 1 -- need km_wine/gmm_wine later for step 5
    km_wine, gmm_wine, km_adult, gmm_adult = run_step1(
        X_wine, y_wine, X_adult, y_adult)

    # step 2 -- dr_models reused in step 4, dr_data reused in step 3
    dr_models, dr_data = run_step2(X_wine, y_wine, X_adult, y_adult)

    run_step3(dr_data, y_wine, y_adult)

    run_step4(X_wine, y_wine, dr_models)

    run_step5(X_wine, y_wine, km_wine, gmm_wine)

    print("\n" + "=" * 60)
    print("done. figures in output/figures/, results in output/results/")
    print("=" * 60)


if __name__ == "__main__":
    main()