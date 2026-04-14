"""
Central place for all path constants and the global random seed.

What this does:

- Defines where the data files live (data/ folder)
- Defines where all outputs go (output/figures/ and output/results/)
- Makes sure those output folders exsit when this module is imported
- Keeps the random seed in one place so every file uses the same one
  instead of each file hardcoding its own seed which would be a mess

Every other module in src/ imports from here instead of hardcoding paths.
That way if we move the project we only need to chage one file not ten.
"""

from pathlib import Path  # file and folder paths as smart objects

ROOT_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT_DIR / "data"
FIGURES_DIR  = ROOT_DIR / "output" / "figures"
RESULTS_DIR  = ROOT_DIR / "output" / "results"
RANDOM_STATE = 42

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)