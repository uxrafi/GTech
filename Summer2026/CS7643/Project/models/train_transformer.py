import argparse
import json
import os
import random
import sys
from typing import Dict

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch import nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "evaluation"))

from figures import plot_loss_curves, plot_pr_curve, plot_roc_and_confusion
from metrics import compute_metrics, format_table3
from models.common import (
    infer_scores,
    load_metadata,
    load_split,
    make_loader,
    pos_weight_from_labels,
    resolve_device,
)
from models.transformer_model import LogAnomalyTransformer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []

    for xb, yb, lb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        lb = lb.to(device)

        logits = model(xb, lb).squeeze(-1)
        loss = criterion(logits, yb)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def collect_scores(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels = []
    scores = []
    for xb, yb, lb in loader:
        xb = xb.to(device)
        lb = lb.to(device)
        logits = model(xb, lb).squeeze(-1)
        scores.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(yb.numpy())
    return np.concatenate(labels).astype(int), np.concatenate(scores)


def save_artifacts(results_dir: str, history: Dict, y_true: np.ndarray, y_score: np.ndarray):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    np.save(os.path.join(results_dir, "y_true.npy"), y_true)
    np.save(os.path.join(results_dir, "y_score.npy"), y_score)


def write_figures(results_dir: str, history: Dict, y_true: np.ndarray, y_score: np.ndarray):
    plots_dir = os.path.join(results_dir, "plots")
    plot_loss_curves(history, plots_dir)
    plot_roc_and_confusion(y_true, y_score, plots_dir)
    plot_pr_curve(y_true, y_score, plots_dir)


def main():
    ap = argparse.ArgumentParser(description="train Transformer log anomaly classifier")
    ap.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "configs", "transformer_config.yaml"),
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    data_dir = cfg["data_dir"]
    checkpoint_dir = cfg["checkpoint_dir"]
    results_dir = cfg["results_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    metadata = load_metadata(data_dir)
    X_train, y_train, len_train = load_split(data_dir, "train")
    X_val, y_val, len_val = load_split(data_dir, "val")
    X_test, y_test, len_test = load_split(data_dir, "test")

    device = resolve_device(cfg.get("device", "auto"))
    print(f"device: {device}")
    print(f"train={len(y_train)} val={len(y_val)} test={len(y_test)} anomalies={int(y_train.sum())}")

    train_loader = make_loader(X_train, y_train, len_train, cfg["batch_size"], shuffle=True)
    val_loader = make_loader(X_val, y_val, len_val, cfg["batch_size"], shuffle=False)

    model = LogAnomalyTransformer(
        vocab_size=cfg.get("vocab_size", metadata["vocab_size"]),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        max_len=cfg.get("max_len", metadata["max_len"]),
        pad_id=cfg.get("pad_id", metadata["pad_id"]),
    ).to(device)

    pos_weight = (
        pos_weight_from_labels(y_train)
        if cfg.get("pos_weight") == "auto"
        else float(cfg["pos_weight"])
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_roc_auc": []}
    best_score = -1.0
    best_path = os.path.join(checkpoint_dir, "best_transformer.pt")

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        _, val_scores = collect_scores(model, val_loader, device)
        val_auc = roc_auc_score(y_val, val_scores)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_roc_auc"].append(val_auc)

        print(
            f"epoch {epoch:02d} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_roc_auc={val_auc:.4f}",
            flush=True,
        )

        if val_auc > best_score:
            best_score = val_auc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "val_roc_auc": val_auc,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    print(f"loaded best checkpoint from epoch {checkpoint['epoch']} (val_roc_auc={checkpoint['val_roc_auc']:.4f})")

    y_score = infer_scores(
        model,
        X_test,
        len_test,
        device,
        batch_size=cfg.get("inference_batch_size", 1024),
    )
    save_artifacts(results_dir, history, y_test, y_score)

    metrics = compute_metrics(y_test, y_score)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    write_figures(results_dir, history, y_test, y_score)
    print(format_table3(metrics))
    print(f"\nartifacts written to {results_dir}")
    print(f"checkpoint: {best_path}")


if __name__ == "__main__":
    main()
