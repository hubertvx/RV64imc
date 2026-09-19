#!/usr/bin/env python3
"""
int8 quantize every weight layer of the litert deit-tiny and check integer math

For each CONV_2D / FULLY_CONNECTED layer (50 in total):
  weights => int8, one scale per output channel (symmetric)
  input  => int8, one scale per layer, from the largest val seen on the calibration images
  bias  => int32, scaled by input_scale * weight_scale
  requantization: the int32 sum is scaled back to int8 with an integer multiplier and shift no floating point
The check runs each quantized layer on images NOT used for calibration, feeding it the FP32
model's real layer input, and reports how close the dequantized int8 output is to the FP32 output.
"""

import argparse
import glob
import math
import os
import re

import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter, OpResolverType

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # average of the red, green and blue channels
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) # spread of each channel
WEIGHT_OPS = ("CONV_2D", "FULLY_CONNECTED")
PERCENTILE = 99.99   # activation range = this percentile of x (ignores single outliers)
N_EVAL = 16          # last images in the folder are kept aside for the error check


def preprocess(path):
    """Resize shorter side to 256, center-crop 224, normalize, NCHW float32."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    w, h = img.size
    l, t = (w - 224) // 2, (h - 224) // 2
    x = (np.asarray(img.crop((l, t, l + 224, t + 224)), dtype=np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[None].copy()


class Model:
    def __init__(self, path):
        self.it = Interpreter(model_path=path, experimental_preserve_all_tensors=True,
                              experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        self.it.allocate_tensors()
        self.tensors = {d["index"]: d for d in self.it.get_tensor_details()}
        self.inp = self.it.get_input_details()[0]["index"]
        self.layers = []                      # one dict per weight layer
        for op in self.it._get_ops_details():
            if op["op_name"] in WEIGHT_OPS:
                i = op["inputs"]
                name = self.tensors[i[1]]["name"]
                name = name[2:] if name.startswith("p_") else name
                self.layers.append({
                    "name": name[:-7] if name.endswith("_weight") else name,
                    "op": op["op_name"],
                    "in": i[0], "w": i[1], "b": i[2] if len(i) > 2 and i[2] >= 0 else None,
                    "out": op["outputs"][0],
                })

    def run(self, x):
        self.it.set_tensor(self.inp, x)
        self.it.invoke()

    def get(self, idx):
        return self.it.get_tensor(idx).copy()


def as_matrix(t, op):
    """Layer input -> [M, K] matrix. CONV_2D (16x16 stride 16, NHWC) is turned into patches."""
    if op == "CONV_2D":
        _, h, w, c = t.shape
        return t.reshape(h // 16, 16, w // 16, 16, c).transpose(0, 2, 1, 3, 4).reshape(-1, 16 * 16 * c)
    return t.reshape(-1, t.shape[-1])


def weight_matrix(w):
    return w.reshape(w.shape[0], -1)          # [N, K]; conv filter [N, kh, kw, cin] flattens in patch order


def quantize_multiplier(m):
    """real multiplier -> (int32 mult in [2^30, 2^31), right shift) with m = mult * 2^-31 * 2^-shift."""
    frac, exp = math.frexp(m)                 # m = frac * 2^exp, frac in [0.5, 1)
    mult = int(round(frac * (1 << 31)))
    if mult == (1 << 31):
        mult //= 2
        exp += 1
    return mult, -exp


def srdhm(acc, mult):
    """saturating rounding doubling high multiply (gemmlowp), int64 arrays."""
    ab = acc.astype(np.int64) * mult
    ab = ab + np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    return np.where(ab >= 0, ab >> 31, -((-ab) >> 31))     # C-style truncating division by 2^31


def rounding_shift(x, shift):
    """rounding arithmetic right shift by `shift` (per output channel, broadcast over rows)."""
    mask = (np.int64(1) << shift) - 1
    rem = x & mask
    thr = (mask >> 1) + (x < 0)
    return (x >> shift) + (rem > thr)


def requantize(acc, mult, shift):
    return np.clip(rounding_shift(srdhm(acc, mult), shift), -128, 127).astype(np.int8)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(here, "..", "models", "model.tflite"))
    ap.add_argument("--calib", default=os.path.join(here, "..", "data", "calib"))
    ap.add_argument("--out", default=os.path.join(here, "..", "models", "deit_tiny_int8.npz"))
    args = ap.parse_args()

    images = sorted(glob.glob(os.path.join(args.calib, "*.jpg")))
    if len(images) <= N_EVAL:
        raise SystemExit(f"need more than {N_EVAL} images in {args.calib}; run scripts/get_calib_images.py")
    calib, evalset = images[:-N_EVAL], images[-N_EVAL:]
    model = Model(args.model)
    L = model.layers
    print(f"{len(L)} weight layers; calibrating on {len(calib)} images, checking on {len(evalset)} others")

    #  Calibration: largest value (at `percentile`) of each layer's input and output over the images.
    amax_in = np.zeros(len(L))
    amax_out = np.zeros(len(L))
    for p in calib:
        model.run(preprocess(p))
        for j, l in enumerate(L):
            amax_in[j] = max(amax_in[j], np.percentile(np.abs(model.get(l["in"])), PERCENTILE))
            amax_out[j] = max(amax_out[j], np.percentile(np.abs(model.get(l["out"])), PERCENTILE))

    #  quantize weights, bias and requantization parameters.
    q = {}
    for j, l in enumerate(L):
        w = weight_matrix(model.get(l["w"]))                              # [N, K] float32
        b = model.get(l["b"]) if l["b"] is not None else np.zeros(w.shape[0], np.float32)
        s_in, s_out = amax_in[j] / 127, amax_out[j] / 127
        s_w = np.abs(w).max(axis=1) / 127                                  # per output channel
        w_q = np.clip(np.round(w / s_w[:, None]), -127, 127).astype(np.int8)
        b_q = np.round(b / (s_in * s_w)).astype(np.int32)
        ms = [quantize_multiplier(s_in * sw / s_out) for sw in s_w]
        mult = np.array([m for m, _ in ms], dtype=np.int32)
        shift = np.array([s for _, s in ms], dtype=np.int32)
        if shift.min() < 0:
            raise SystemExit(f"{l['name']}: real multiplier >= 1 (needs a left shift); not supported yet")
        q[l["name"]] = dict(w_q=w_q, b_q=b_q, mult=mult, shift=shift, s_in=s_in, s_out=s_out)

    # check integer layer vs FP32 layer on images not used for calibration.
    err = {l["name"]: [] for l in L}
    acc_max = 0                       # largest int32 accumulator
    for p in evalset:
        model.run(preprocess(p))
        for l in L:
            d = q[l["name"]]
            x = as_matrix(model.get(l["in"]), l["op"])
            ref = as_matrix(model.get(l["out"]), "FULLY_CONNECTED")
            x_q = np.clip(np.round(x / d["s_in"]), -128, 127).astype(np.int8)
            acc = x_q.astype(np.int32) @ d["w_q"].T.astype(np.int32) + d["b_q"]   # int32 accumulate
            acc_max = max(acc_max, int(np.abs(acc).max()))
            out = requantize(acc, d["mult"].astype(np.int64), d["shift"].astype(np.int64)).astype(np.float32) * d["s_out"]
            err[l["name"]].append(np.linalg.norm(out - ref) / np.linalg.norm(ref))

    # One row per layer kind 
    kinds = {}
    for l in L:
        kinds.setdefault(re.sub(r"blocks_\d+_", "", l["name"]), []).append(l["name"])
    print(f"\n{'layer kind':<18}{'layers':>7}{'K':>6}{'N':>6}{'mean err':>10}{'worst err':>11}")
    for kind, names in kinds.items():
        n, k = q[names[0]]["w_q"].shape
        e = [np.mean(err[nm]) for nm in names]
        print(f"{kind:<18}{len(names):>7}{k:>6}{n:>6}{np.mean(e):>10.4f}{max(e):>11.4f}")
    print(f"\nint8 weights: {sum(d['w_q'].size for d in q.values()):,} bytes")
    print(f"largest |accumulator|: {acc_max:,}  (needs {acc_max.bit_length() + 1} signed bits)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **{f"{n}__{k}": v for n, d in q.items() for k, v in d.items()})
    print(f"Wrote {os.path.normpath(args.out)}")


if __name__ == "__main__":
    main()
