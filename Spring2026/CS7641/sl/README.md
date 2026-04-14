# CS7641 Supervised Learning - Spring 2026

This repository contains the code and analysis for the supervised learning report on the Adult Income and Wine Quality datasets.

## Folder Structure

- `data/` – contains the raw CSV files (`adult.csv`, `wine.csv`).
- `outputs/` – generated figures and JSON result files.
- `src/` – Python modules for data loading, modeling, plotting, and utilities.
    - `__init__.py`
    - `data_loader.py`
    - `inspect_data.py` – quick data inspection script.
    - `models.py`
    - `paths.py` – central configuration of paths and random seed.
    - `plotting.py`
    - `run_analysis.py` – main script to run the entire experiment.
    - `utils.py`

- `requirements.txt` – list of required Python packages.

## Setup

1. Place the dataset files in the `data/` folder.
2. Install dependencies:

```bash
pip install -r requirements.txt