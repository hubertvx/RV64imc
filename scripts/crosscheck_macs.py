#!/usr/bin/env python3
"""Independent cross-check of the MAC count using fvcore on the PyTorch (timm) DeiT-Tiny

The LiteRT file was converted from this timm model, so the
matmul/linear/conv MACs must equal scripts/analyze_workload.py. fvcore reports MACs under the name
"flops", and also counts LayerNorm, which is not a matmul, so that entry is shown separately.
"""
import timm
import torch
from fvcore.nn import FlopCountAnalysis

MATMUL_LIKE = ("conv", "linear", "matmul")


def main():
    model = timm.create_model("deit_tiny_patch16_224", pretrained=False).eval()
    for mod in model.modules():
        if hasattr(mod, "fused_attn"):
            mod.fused_attn = False

    counter = FlopCountAnalysis(model, torch.randn(1, 3, 224, 224))
    counter.unsupported_ops_warnings(False)
    counter.uncalled_modules_warnings(False)
    by_op = counter.by_operator()

    for name, macs in by_op.items():
        print(f"{name:<12}{macs:>16,}")
    matmul_total = sum(v for k, v in by_op.items() if k in MATMUL_LIKE)
    print(f"{'\nTotal':<12}{matmul_total:>16,}   ({matmul_total / 1e9:.3f} GMAC)")


if __name__ == "__main__":
    main()
