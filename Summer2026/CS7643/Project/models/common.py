import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_metadata(data_dir: str) -> Dict:
    with open(os.path.join(data_dir, "metadata.json")) as f:
        return json.load(f)


def load_split(data_dir: str, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.load(os.path.join(data_dir, f"X_{split}.npy"))
    y = np.load(os.path.join(data_dir, f"y_{split}.npy"))
    lengths = np.load(os.path.join(data_dir, f"len_{split}.npy"))
    return X, y, lengths


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(X, dtype=torch.long),
        torch.as_tensor(y, dtype=torch.float32),
        torch.as_tensor(lengths, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def pos_weight_from_labels(y: np.ndarray) -> float:
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0:
        raise ValueError("training split has zero positive examples")
    return n_neg / n_pos


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


@torch.no_grad()
def infer_scores(
    model: torch.nn.Module,
    X: np.ndarray,
    lengths: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    scores = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i : i + batch_size], dtype=torch.long, device=device)
        lb = torch.as_tensor(lengths[i : i + batch_size], dtype=torch.long, device=device)
        logits = model(xb, lb).squeeze(-1)
        scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)
