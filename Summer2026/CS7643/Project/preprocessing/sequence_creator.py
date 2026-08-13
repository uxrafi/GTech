"""build the token vocabulary and turn event sequences into padded id arrays.

vocab contract: {"<PAD>": 0, "<UNK>": 1, "E1": 2, "E2": 3, ...}. sequences are
truncated/right-padded to a fixed length T; true (clipped) lengths are kept so
the model can pack_padded_sequence.
"""
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1


def build_vocab(templates_path: str) -> Dict[str, int]:
    """map event templates -> contiguous ids, ordered by template number."""
    df = pd.read_csv(templates_path, usecols=["EventId"])
    event_ids = sorted(df["EventId"].tolist(), key=lambda e: int(re.sub(r"\D", "", e)))
    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for event in event_ids:
        vocab[event] = len(vocab)
    return vocab


def encode_and_pad(
    traces: List[Tuple[str, List[str]]],
    vocab: Dict[str, int],
    max_len: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """encode tokens -> ids, truncate/right-pad to max_len.

    returns (X [N, max_len] int32, lengths [N] int32, block_ids, n_truncated).
    """
    n = len(traces)
    X = np.full((n, max_len), PAD_ID, dtype=np.int32)
    lengths = np.zeros(n, dtype=np.int32)
    block_ids: List[str] = []
    n_truncated = 0

    for i, (block_id, seq) in enumerate(traces):
        block_ids.append(block_id)
        if len(seq) > max_len:
            n_truncated += 1
        clipped = seq[:max_len]
        lengths[i] = len(clipped)
        for j, token in enumerate(clipped):
            X[i, j] = vocab.get(token, UNK_ID)

    return X, lengths, block_ids, n_truncated
