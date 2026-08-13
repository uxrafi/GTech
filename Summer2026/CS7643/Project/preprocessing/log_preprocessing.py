"""end-to-end preprocessing: raw traces -> train/val/test .npy + data contract.

reproducibility deliverable: a fresh clone with the raw csvs in data/raw and

    python preprocessing/log_preprocessing.py --raw data/raw --out data/sequences

regenerates every array and json in data/sequences.
"""
import argparse
import json
import os
import sys

import numpy as np

# allow running as a plain script (python preprocessing/log_preprocessing.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from labeler import attach_labels, load_labels, stratified_split
from log_parser import parse_event_traces
from sequence_creator import build_vocab, encode_and_pad


def save_split(out_dir, name, X, y, lengths, idx):
    np.save(os.path.join(out_dir, f"X_{name}.npy"), X[idx])
    np.save(os.path.join(out_dir, f"y_{name}.npy"), y[idx])
    np.save(os.path.join(out_dir, f"len_{name}.npy"), lengths[idx])


def main():
    ap = argparse.ArgumentParser(description="HDFS log preprocessing pipeline")
    ap.add_argument("--raw", default="data/raw", help="dir with the raw csvs")
    ap.add_argument("--out", default="data/sequences", help="output dir")
    ap.add_argument("--max-len", type=int, default=50, help="fixed sequence length T")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    args = ap.parse_args()

    traces_path = os.path.join(args.raw, "Event_traces.csv")
    labels_path = os.path.join(args.raw, "anomaly_label.csv")
    templates_path = os.path.join(args.raw, "HDFS.log_templates.csv")
    os.makedirs(args.out, exist_ok=True)

    print(f"[1/5] parsing traces from {traces_path}")
    traces = parse_event_traces(traces_path)

    print(f"[2/5] building vocab from {templates_path}")
    vocab = build_vocab(templates_path)

    print(f"[3/5] encoding + padding to T={args.max_len}")
    X, lengths, block_ids, n_truncated = encode_and_pad(traces, vocab, args.max_len)

    print(f"[4/5] labeling from {labels_path}")
    label_map = load_labels(labels_path)
    y = attach_labels(block_ids, label_map)

    print("[5/5] stratified split + save")
    train_idx, val_idx, test_idx = stratified_split(
        y, args.seed, args.val_frac, args.test_frac
    )
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    for name, idx in splits.items():
        save_split(args.out, name, X, y, lengths, idx)

    with open(os.path.join(args.out, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)

    block_arr = np.array(block_ids)
    split_ids = {name: block_arr[idx].tolist() for name, idx in splits.items()}
    with open(os.path.join(args.out, "splits.json"), "w") as f:
        json.dump(split_ids, f)

    class_counts = {
        name: {"normal": int((y[idx] == 0).sum()), "anomaly": int((y[idx] == 1).sum())}
        for name, idx in splits.items()
    }
    metadata = {
        "n_total": len(y),
        "max_len": args.max_len,
        "vocab_size": len(vocab),
        "seed": args.seed,
        "split_ratios": {
            "train": round(1 - args.val_frac - args.test_frac, 4),
            "val": args.val_frac,
            "test": args.test_frac,
        },
        "class_counts": class_counts,
        "n_truncated": n_truncated,
        "pct_truncated": round(100 * n_truncated / len(y), 4),
        "pad_id": 0,
        "unk_id": 1,
        "source": "LogHub HDFS_v1 Event_traces.csv (Drain-parsed upstream)",
    }
    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print_summary(args.out, metadata, class_counts)


def print_summary(out_dir, metadata, class_counts):
    print("\n=== preprocessing summary ===")
    print(f"output dir : {out_dir}")
    print(f"total      : {metadata['n_total']} blocks")
    print(f"vocab_size : {metadata['vocab_size']}   max_len: {metadata['max_len']}   seed: {metadata['seed']}")
    print(f"truncated  : {metadata['n_truncated']} ({metadata['pct_truncated']}%)")
    print(f"{'split':<8}{'total':>10}{'normal':>10}{'anomaly':>10}{'anom %':>10}")
    for name, c in class_counts.items():
        total = c["normal"] + c["anomaly"]
        pct = 100 * c["anomaly"] / total if total else 0
        print(f"{name:<8}{total:>10}{c['normal']:>10}{c['anomaly']:>10}{pct:>9.2f}%")


if __name__ == "__main__":
    main()
