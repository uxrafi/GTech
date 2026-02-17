"""
Centralized paths and constants for the project.

What this does:

- Defines base directorys for data, outputs, and figures
- Auto-creates directorys if they don't exist yet
- Sets global random seed for reproducability
- Prints debug info to confirm paths are correct

All other modules import from here to avoid hardcoding paths everywhere.
"""

##########################

"""
Assignment Requirements Covered:

- Fixed random seed (RANDOM_STATE = 42) for reproducability
- Centralized directory structure for outputs and figures
- Deterministic path handling across all modules
- Supports "Methodology & reproducability" requirement
- Auto-creation of necesary directories
"""

import os  # OS-level operatons (not actualy used here but kept for compatability)
from pathlib import Path  # modern way to handle file paths

# Figure out where this file lives, then go up one level to get projet root
BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_DIR / 'data'  # where CSVs live
OUTPUT_DIR = BASE_DIR / 'outputs'  # where results get saved
FIGURES_DIR = OUTPUT_DIR / 'figures'  # where plots go

# Debug print to confirm paths are setup corectly
# Helps catch issues if runing from wrong directory
print(f"DEBUG: DATA_DIR = {DATA_DIR}")

# Create directorys if they don't exist yet
# parents=True means create intermediate directorys too
# exist_ok=True means don't error if directory allready exists
for d in [DATA_DIR, OUTPUT_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Global random seed for reproducability
# Fixed at 42 so experments give same results every time
RANDOM_STATE = 42