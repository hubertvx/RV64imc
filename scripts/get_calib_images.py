#!/usr/bin/env python3
"""Save a few natural images, and labels to data/calib/ for calibration and error checks.

imagenet : ILSVRC/imagenet-1k on Hugging Face.
"""
import argparse
import csv
import os

from datasets import load_dataset

SOURCES = {
    "imagenet": "ILSVRC/imagenet-1k",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=SOURCES, default="imagenet")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--count", type=int, default=64)
    ap.add_argument("--stride", type=int, default=50, help="take every stride-th image")
    ap.add_argument("--offset", type=int, default=0, help="index of the first image taken (< stride)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "calib"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stream = load_dataset(SOURCES[args.source], split=args.split, streaming=True)
    rows = []
    for i, ex in enumerate(stream):
        if len(rows) >= args.count:
            break
        if i % args.stride != args.offset:
            continue
        name = f"{len(rows):03d}.jpg"
        ex["image"].convert("RGB").save(os.path.join(args.out, name), quality=95)
        rows.append((name, ex["label"]))
    with open(os.path.join(args.out, "labels.csv"), "w", newline="") as f:
        csv.writer(f).writerows([("file", "label")] + rows)
    classes = len({r[1] for r in rows})
    if classes < min(10, len(rows)):
        print(f"WARNING: only {classes} distinct classes; increase --stride")
    print(f"saved {len(rows)} {args.source}/{args.split} images to {os.path.normpath(args.out)}")


if __name__ == "__main__":
    main()
