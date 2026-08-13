import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "evaluation"))

from figures import plot_loss_curves, plot_pr_curve, plot_roc_and_confusion
from metrics import compute_metrics, format_table3
from models.common import infer_scores, load_metadata, load_split, resolve_device
from models.transformer_model import LogAnomalyTransformer


def build_model(cfg: dict, metadata: dict) -> LogAnomalyTransformer:
    return LogAnomalyTransformer(
        vocab_size=cfg.get("vocab_size", metadata["vocab_size"]),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        max_len=cfg.get("max_len", metadata["max_len"]),
        pad_id=cfg.get("pad_id", metadata["pad_id"]),
    )


def main():
    ap = argparse.ArgumentParser(description="evaluate Transformer checkpoint on test split")
    ap.add_argument("--checkpoint", default="checkpoints/best_transformer.pt")
    ap.add_argument("--data-dir", default="sequences")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--history", default=None, help="optional history.json for loss curves")
    args = ap.parse_args()

    device = resolve_device("auto")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    metadata = load_metadata(args.data_dir)

    model = build_model(cfg, metadata).to(device)
    model.load_state_dict(checkpoint["model_state"])

    _, y_test, len_test = load_split(args.data_dir, "test")
    X_test, _, _ = load_split(args.data_dir, "test")

    y_score = infer_scores(
        model,
        X_test,
        len_test,
        device,
        batch_size=cfg.get("inference_batch_size", 1024),
    )

    os.makedirs(args.results_dir, exist_ok=True)
    np.save(os.path.join(args.results_dir, "y_true.npy"), y_test)
    np.save(os.path.join(args.results_dir, "y_score.npy"), y_score)

    metrics = compute_metrics(y_test, y_score)
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plots_dir = os.path.join(args.results_dir, "plots")
    history_path = args.history or os.path.join(args.results_dir, "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_loss_curves(history, plots_dir)
    plot_roc_and_confusion(y_test, y_score, plots_dir)
    plot_pr_curve(y_test, y_score, plots_dir)

    print(format_table3(metrics))


if __name__ == "__main__":
    main()
