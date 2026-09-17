#!/usr/bin/env python3
"""
Count the multiply-accumulates (MACs) of every matmul-like operator in the LiteRT DeiT-Tiny graph.

The table gives each operator's share of the total and whether K is a multiple of 8 (needed to pack 8 int8 lanes into one 64-bit register for a packed MAC instruction).
"""

import argparse
import csv
import os
import re
from collections import OrderedDict

from ai_edge_litert.interpreter import Interpreter, OpResolverType

MAC_OPS = ("CONV_2D", "FULLY_CONNECTED", "BATCH_MATMUL")

def prod(shape):
    p = 1
    for s in shape:
        p *= s
    return p


def label(op_name, weight_name, shape):
    """Name an operator from the model's own tensor names, e.g. 'p_blocks_3_mlp_fc1_weight' -> 'mlp_fc1'.

    BATCH_MATMUL has no weight tensor (both operands are activations), so it is named by its shape.
    """
    if op_name == "BATCH_MATMUL":
        return "matmul_%dx%dx%d" % shape
    name = re.sub(r"^p_|_weight$", "", weight_name)
    return re.sub(r"blocks_\d+_", "", name)


def mac_shape(op, shapes_in, shape_out):
    """Return (M, N, K, count) for one matmul-like operator."""
    if op == "CONV_2D":                       # input NHWC, filter [Cout, kh, kw, Cin]
        _, h, w, cout = shape_out
        _, kh, kw, cin = shapes_in[1]
        return h * w, cout, kh * kw * cin, 1
    if op == "FULLY_CONNECTED":               # input [..., K], weights [N, K]
        n, k = shapes_in[1]
        return prod(shapes_in[0][:-1]), n, k, 1
    # BATCH_MATMUL: output [..., M, N]; K follows from the left operand's size.
    m, n = shape_out[-2], shape_out[-1]
    count = prod(shape_out[:-2])
    return m, n, prod(shapes_in[0]) // (count * m), count


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(here, "..", "models", "model.tflite"))
    ap.add_argument("--csv", help="write the per-operator table to this CSV file")
    args = ap.parse_args()

    it = Interpreter(model_path=args.model,
                     experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    tensors = {d["index"]: d for d in it.get_tensor_details()}

    groups = OrderedDict()
    for op in it._get_ops_details():
        if op["op_name"] not in MAC_OPS:
            continue
        shapes_in = [tuple(int(s) for s in tensors[i]["shape"]) for i in op["inputs"] if i >= 0]
        shape_out = tuple(int(s) for s in tensors[op["outputs"][0]]["shape"])
        m, n, k, cnt = mac_shape(op["op_name"], shapes_in, shape_out)
        role = label(op["op_name"], tensors[op["inputs"][1]]["name"], (m, n, k))
        g = groups.setdefault(role, {"op": op["op_name"], "shape": (m, n, k), "count": 0})
        g["count"] += cnt

    total = sum(prod(g["shape"]) * g["count"] for g in groups.values())
    print(f"{'operator':<20}{'LiteRT op':<17}{'M':>6}{'N':>6}{'K':>6}{'count':>7}{'MACs':>16}{'share':>8}  K%8==0")
    rows = []
    for role, g in groups.items():
        m, n, k = g["shape"]
        macs = m * n * k * g["count"]
        ok = "yes" if k % 8 == 0 else "NO"
        print(f"{role:<20}{g['op']:<17}{m:>6}{n:>6}{k:>6}{g['count']:>7}{macs:>16,}{100 * macs / total:>7.1f}%  {ok}")
        rows.append([role, g["op"], m, n, k, g["count"], macs, ok])
    print(f"{'\nTOTAL':<37}{'':>25}{total:>16,}   ({total / 1e9:.3f} GMAC)")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["operator", "litert_op", "M", "N", "K", "count", "macs", "k_mult_of_8"])
            w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
