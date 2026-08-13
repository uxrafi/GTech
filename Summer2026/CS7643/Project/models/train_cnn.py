"""train and evaluate the 1-D CNN anomaly detector.

usage:
    python models/train_cnn.py
    python models/train_cnn.py --seq data/sequences --out results/cnn --epochs 20
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "evaluation"))

from metrics import compute_metrics, format_table3
from figures import plot_loss_curves, plot_pr_curve, plot_roc_and_confusion
from cnn1d_model import CNN1DClassifier


def load_split(seq_dir, split):
    X = np.load(os.path.join(seq_dir, f"X_{split}.npy")).astype(np.int64)
    y = np.load(os.path.join(seq_dir, f"y_{split}.npy")).astype(np.float32)
    return X, y


def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            n += len(yb)
    return total_loss / n


def main():
    ap = argparse.ArgumentParser(description="train 1-D CNN on HDFS sequences")
    ap.add_argument("--seq", default="data/sequences", help="dir with .npy splits")
    ap.add_argument("--out", default="results/cnn", help="output dir for model + metrics")
    ap.add_argument("--plots-out", default="results/plots/cnn", help="figure output dir")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embedding-dim", type=int, default=64)
    ap.add_argument("--num-filters", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"loading sequences from {args.seq}")
    X_train, y_train = load_split(args.seq, "train")
    X_val,   y_val   = load_split(args.seq, "val")
    X_test,  y_test  = load_split(args.seq, "test")

    with open(os.path.join(args.seq, "vocab.json")) as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    print(f"vocab_size={vocab_size}  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   args.batch_size, shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  args.batch_size, shuffle=False)

    model = CNN1DClassifier(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        num_filters=args.num_filters,
        dropout=args.dropout,
    ).to(device)
    print(model)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    print(f"pos_weight={pos_weight.item():.2f}  (n_pos={n_pos}, n_neg={n_neg})")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        tr_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl_loss = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        print(f"epoch {epoch:3d}  train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}")
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), ckpt_path)

    print(f"\nloading best checkpoint from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_scores = []
    with torch.no_grad():
        for xb, _ in test_loader:
            logits = model(xb.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(probs)
    y_score = np.concatenate(all_scores)

    metrics = compute_metrics(y_test, y_score)
    print()
    print(format_table3(metrics))

    np.save(os.path.join(args.out, "y_score.npy"), y_score)
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"hyperparameters": vars(args), "test": metrics}, f, indent=2)

    os.makedirs(args.plots_out, exist_ok=True)
    made = [
        plot_loss_curves(history, args.plots_out),
        plot_roc_and_confusion(y_test, y_score, args.plots_out),
        plot_pr_curve(y_test, y_score, args.plots_out),
    ]
    print("\nfigures written:")
    for p in made:
        print(f"  {p}")


if __name__ == "__main__":
    main()