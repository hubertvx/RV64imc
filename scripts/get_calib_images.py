#!/usr/bin/env python3
"""
Download the calibration and evaluation images (COCO val2017 subset) listed in data/coco_manifest.csv.

COCO val2017

Two disjoint sets are used, drawn with a fixed seed:
  calib  256 images   only these set the activation scales (scripts/quantize_w8a8.py)
  eval  1000 images   never used for calibration; the layer-error check and, later, the accuracy measurement
"""

import argparse
import csv
import hashlib
import io
import json
import os
import random
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
MANIFEST = os.path.join(DATA, "coco_manifest.csv")
OUT = os.path.join(DATA, "coco")

N_CALIB, N_EVAL, SEED = 256, 1000, 2026
LICENSE_IDS = (4, 5, 7)
COLUMNS = ["split", "file", "sha256", "license", "flickr_url", "coco_url"]


def fetch(url, tries=4):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:
            if attempt == tries - 1:
                raise SystemExit(f"download failed: {url}: {e}")
            time.sleep(2 * (attempt + 1))


def download(row, check):
    path = os.path.join(OUT, row["split"], row["file"])
    if os.path.exists(path) and (
        not check or sha256(open(path, "rb").read()) == row["sha256"]
    ):
        return row, None
    data = fetch(row["coco_url"])
    digest = sha256(data)
    if check and digest != row["sha256"]:
        raise SystemExit(
            f"{row['file']}: sha256 {digest} differs from the manifest ({row['sha256']})"
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return row, digest


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def make_manifest(zip_path):
    d = json.load(
        io.TextIOWrapper(
            zipfile.ZipFile(zip_path).open("annotations/instances_val2017.json")
        )
    )
    names = {l["id"]: l["name"] for l in d["licenses"]}
    pool = sorted(
        (i for i in d["images"] if i["license"] in LICENSE_IDS),
        key=lambda i: i["file_name"],
    )
    if len(pool) < N_CALIB + N_EVAL:
        raise SystemExit(
            f"only {len(pool)} images under the allowed licences, need {N_CALIB + N_EVAL}"
        )
    chosen = random.Random(SEED).sample(pool, N_CALIB + N_EVAL)
    rows = []
    for k, img in enumerate(chosen):
        rows.append(
            dict(
                split="calib" if k < N_CALIB else "eval",
                file=img["file_name"],
                sha256="",
                license=names[img["license"]],
                flickr_url=img["flickr_url"],
                coco_url=img["coco_url"],
            )
        )
    with ThreadPoolExecutor(8) as ex:
        for row, digest in ex.map(lambda r: download(r, check=False), rows):
            row["sha256"] = digest or sha256(
                open(os.path.join(OUT, row["split"], row["file"]), "rb").read()
            )
    rows.sort(key=lambda r: (r["split"], r["file"]))
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(
        f"wrote {os.path.normpath(MANIFEST)}: {N_CALIB} calib + {N_EVAL} eval images, seed {SEED}, "
        f"pool of {len(pool)} images under licences {LICENSE_IDS}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--make-manifest",
        metavar="ANNOTATIONS_ZIP",
        help="rebuild data/coco_manifest.csv from COCO's annotations_trainval2017.zip (maintainers)",
    )
    args = ap.parse_args()
    if args.make_manifest:
        return make_manifest(args.make_manifest)

    with open(MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))
    done = 0
    with ThreadPoolExecutor(8) as ex:
        for _ in ex.map(lambda r: download(r, check=True), rows):
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(rows)}", flush=True)
    n = {s: sum(r["split"] == s for r in rows) for s in ("calib", "eval")}
    print(
        f"ok: {n['calib']} calibration and {n['eval']} evaluation images in {os.path.normpath(OUT)}, "
        f"all sha256 hashes match the manifest"
    )


if __name__ == "__main__":
    main()
