"""join sequences to ground-truth labels and make a stratified split.

labels come from anomaly_label.csv (Normal=0, Anomaly=1). the split is a
stratified 80/10/10 by block with a fixed seed, so the ~2.9% anomaly ratio is
preserved in every split and there is no cross-split leakage (one block = one
sequence).
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_MAP = {"Normal": 0, "Anomaly": 1}


def load_labels(path: str) -> Dict[str, int]:
    """read anomaly_label.csv -> {block_id: 0|1}."""
    df = pd.read_csv(path)
    return {row.BlockId: LABEL_MAP[row.Label] for row in df.itertuples(index=False)}


def attach_labels(block_ids: List[str], label_map: Dict[str, int]) -> np.ndarray:
    """align labels to block order; raise if any block is unlabeled."""
    missing = [b for b in block_ids if b not in label_map]
    if missing:
        raise ValueError(f"{len(missing)} blocks have no label (e.g. {missing[:3]})")
    return np.array([label_map[b] for b in block_ids], dtype=np.int8)


def stratified_split(
    y: np.ndarray,
    seed: int,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """stratified index split -> (train_idx, val_idx, test_idx)."""
    idx = np.arange(len(y))
    holdout = val_frac + test_frac
    train_idx, rest_idx = train_test_split(
        idx, test_size=holdout, stratify=y, random_state=seed
    )
    # split the holdout into val/test, preserving stratification
    val_share = val_frac / holdout
    val_idx, test_idx = train_test_split(
        rest_idx, train_size=val_share, stratify=y[rest_idx], random_state=seed
    )
    return train_idx, val_idx, test_idx
