#!/usr/bin/env python3
"""
Convert The K.I.S.S. NCA .weights.h5 files into compact JSON files readable by the
static GitHub Pages web app.

Example:
  python tools/convert_h5_to_web.py \
    --input "pipeline/1_grow/kiss_log/the_kiss/1200.weights.h5" \
    --output "assets/models/weak.json" \
    --name "Weak model" \
    --target-size 64 \
    --channel-n 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def collect_datasets(h5_path: Path):
    datasets = []
    with h5py.File(h5_path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                arr = np.asarray(obj)
                datasets.append((name, arr))
        f.visititems(visitor)
    return datasets


def find_weights(datasets, channel_n: int, hidden_n: int | None):
    perception_n = channel_n * 3

    w1_candidates = []
    w2_candidates = []
    b1_candidates = []
    b2_candidates = []

    for name, arr in datasets:
        shape = arr.shape
        if arr.ndim == 4 and shape[0] == 1 and shape[1] == 1 and shape[2] == perception_n:
            if hidden_n is None or shape[3] == hidden_n:
                w1_candidates.append((name, arr))
        if arr.ndim == 4 and shape[0] == 1 and shape[1] == 1 and shape[3] == channel_n:
            if hidden_n is None or shape[2] == hidden_n:
                w2_candidates.append((name, arr))
        if arr.ndim == 1 and (hidden_n is None or shape[0] == hidden_n):
            b1_candidates.append((name, arr))
        if arr.ndim == 1 and shape[0] == channel_n:
            b2_candidates.append((name, arr))

    if not w1_candidates:
        raise RuntimeError(f"Could not find first 1x1 Conv2D kernel with input={perception_n}.")

    # Prefer dmodel/sequential layer names if present.
    w1_name, W1 = sorted(w1_candidates, key=lambda x: ("dmodel" not in x[0] and "sequential" not in x[0], x[0]))[0]
    detected_hidden = W1.shape[3]

    w2_candidates = [(n, a) for n, a in w2_candidates if a.shape[2] == detected_hidden]
    b1_candidates = [(n, a) for n, a in b1_candidates if a.shape[0] == detected_hidden]

    if not w2_candidates:
        raise RuntimeError(f"Could not find second 1x1 Conv2D kernel with hidden={detected_hidden}, output={channel_n}.")
    if not b1_candidates:
        raise RuntimeError(f"Could not find first Conv2D bias with hidden={detected_hidden}.")
    if not b2_candidates:
        raise RuntimeError(f"Could not find second Conv2D bias with output={channel_n}.")

    w2_name, W2 = sorted(w2_candidates, key=lambda x: ("dmodel" not in x[0] and "sequential" not in x[0], x[0]))[0]
    b1_name, b1 = sorted(b1_candidates, key=lambda x: ("dmodel" not in x[0] and "sequential" not in x[0], x[0]))[0]
    b2_name, b2 = sorted(b2_candidates, key=lambda x: ("dmodel" not in x[0] and "sequential" not in x[0], x[0]))[0]

    print("Selected datasets:")
    print(f"  W1: {w1_name} {W1.shape}")
    print(f"  b1: {b1_name} {b1.shape}")
    print(f"  W2: {w2_name} {W2.shape}")
    print(f"  b2: {b2_name} {b2.shape}")

    return W1, b1, W2, b2, detected_hidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--name", default="The K.I.S.S. NCA model")
    ap.add_argument("--source", default=None)
    ap.add_argument("--target-size", type=int, default=64)
    ap.add_argument("--channel-n", type=int, default=16)
    ap.add_argument("--hidden-n", type=int, default=128)
    ap.add_argument("--list", action="store_true", help="Only list datasets in the H5 file.")
    args = ap.parse_args()

    datasets = collect_datasets(args.input)

    if args.list:
        for name, arr in datasets:
            print(f"{name}: {arr.shape} {arr.dtype}")
        return

    W1, b1, W2, b2, hidden_n = find_weights(datasets, args.channel_n, args.hidden_n)

    payload = {
        "format": "kiss-nca-web-v1",
        "name": args.name,
        "source": args.source or str(args.input),
        "target_size": args.target_size,
        "channel_n": args.channel_n,
        "hidden_n": int(hidden_n),
        "W1": W1.reshape(args.channel_n * 3, hidden_n).astype(np.float32).ravel().tolist(),
        "b1": b1.astype(np.float32).ravel().tolist(),
        "W2": W2.reshape(hidden_n, args.channel_n).astype(np.float32).ravel().tolist(),
        "b2": b2.astype(np.float32).ravel().tolist(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Wrote {args.output} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
