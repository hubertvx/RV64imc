#!/usr/bin/env python3
"""
Compare kernel configurations side by side and write results/comparison.md.

    python scripts/compare_configs.py b0 x1

The FIRST configuration is the baseline: "reduction" = baseline instructions / this configuration's instructions.

Retired-instruction counts from Spike (ISA level)

"""

import csv
import os
import sys

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
GEMM, REQUANT = "gemm_s8", "requant"
WORK_UNIT = {GEMM: "MAC", REQUANT: "output"}
MIX_CLASSES = [
    ("load", "load"),
    ("store", "store"),
    ("mul", "multiply"),
    ("branch", "branch/jump"),
    ("alu", "other ALU"),
    ("custom", "custom (qmac8)"),
]
GEMM_REDUCTION_TARGET = 2.0


def read_rows(name, key):
    with open(os.path.join(RESULTS, name), newline="") as f:
        return {row[key]: row for row in csv.DictReader(f)}


def number(row, column):
    """A float from a CSV cell, or None if the cell is empty (the instruction mix was not traced)."""
    value = row.get(column, "")
    return float(value) if value != "" else None


def fmt(x, digits=3):
    return "-" if x is None else f"{x:,.{digits}f}"


def count(x):
    return "-" if x is None else f"{int(x):,}"


def markdown_table(headers, rows, left=1):
    """A markdown table with every column padded to the same width, so it also reads well as plain text in a
    terminal. The first `left` columns (names) are left-aligned, the others (numbers) right-aligned.
    """
    cells = [[str(c) for c in row] for row in [headers] + rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]

    def line(row):
        return (
            "| "
            + " | ".join(
                c.ljust(w) if i < left else c.rjust(w)
                for i, (c, w) in enumerate(zip(row, widths))
            )
            + " |"
        )

    separator = (
        "|"
        + "|".join(
            (":" + "-" * (w + 1)) if i < left else ("-" * (w + 1) + ":")
            for i, w in enumerate(widths)
        )
        + "|"
    )
    return [line(cells[0]), separator] + [line(row) for row in cells[1:]]


def main():
    configs = sys.argv[1:]
    if len(configs) < 2:
        sys.exit(
            "usage: compare_configs.py <baseline> <config> [<config> ...]   e.g. compare_configs.py b0 x1"
        )
    base = configs[0]
    kernels = {c: read_rows(f"{c}.csv", "kernel") for c in configs}
    operators = {c: read_rows(f"{c}_operators.csv", "operator") for c in configs}
    traced = all(number(kernels[c][GEMM], "load_per_work") is not None for c in configs)

    out = []

    others = configs[1:]

    # headline: the numbers for a slide
    out += [f"### Headline: {base} vs {', '.join(others)}", ""]
    gemm_instr = {c: number(kernels[c][GEMM], "instret") for c in configs}
    total_instr = {
        c: gemm_instr[c] + number(kernels[c][REQUANT], "instret") for c in configs
    }
    mac_per_instr = {c: number(kernels[c][GEMM], "mac_per_instr") for c in configs}
    requant_share = {
        c: 100 * number(kernels[c][REQUANT], "instret") / total_instr[c]
        for c in configs
    }
    rows = [
        ["GEMM retired instructions"]
        + [count(gemm_instr[c]) for c in configs]
        + [f"{gemm_instr[base] / gemm_instr[c]:.2f}x fewer" for c in others],
        ["GEMM instructions per MAC"]
        + [fmt(number(kernels[c][GEMM], "instr_per_work")) for c in configs]
        + ["" for _ in others],
        ["MACs per retired instruction"]
        + [fmt(mac_per_instr[c], 4) for c in configs]
        + [f"{mac_per_instr[c] / mac_per_instr[base]:.2f}x more" for c in others],
        ["Total, GEMM + requantization"]
        + [count(total_instr[c]) for c in configs]
        + [f"{total_instr[base] / total_instr[c]:.2f}x fewer" for c in others],
        ["Requantization share of total"]
        + [f"{requant_share[c]:.1f}%" for c in configs]
        + ["" for _ in others],
    ]
    out += markdown_table(
        ["metric"] + configs + [f"{c} vs {base}" for c in others], rows
    )

    # instruction counts per kernel
    out += [
        "",
        "### Retired instructions per kernel (work item = MAC for gemm_s8, output element for requant)",
        "",
    ]
    rows = []
    for kernel in (GEMM, REQUANT):
        for c in configs:
            r = kernels[c][kernel]
            reduction = number(kernels[base][kernel], "instret") / number(r, "instret")
            rows.append(
                [
                    kernel,
                    c,
                    count(number(r, "instret")),
                    f"{reduction:.2f}x",
                    fmt(number(r, "instr_per_work")),
                    fmt(number(r, "mac_per_instr"), 4) if kernel == GEMM else "-",
                    count(number(r, "kernel_code_bytes")),
                ]
            )
    out += markdown_table(
        [
            "kernel",
            "config",
            "retired instr",
            "reduction",
            "instr/work",
            "MAC/instr",
            "code bytes",
        ],
        rows,
        left=2,
    )

    # instruction mix per work item
    out += ["", "### Instruction mix per work item (traced on the 2-layer sample)", ""]
    rows = []
    for kernel in (GEMM, REQUANT):
        for c in configs:
            r = kernels[c][kernel]
            rows.append(
                [kernel, c]
                + [fmt(number(r, f"{key}_per_work"), 4) for key, _ in MIX_CLASSES]
                + [fmt(number(r, "ldst_per_work"), 4)]
            )
    out += markdown_table(
        ["kernel", "config"] + [label for _, label in MIX_CLASSES] + ["load+store"],
        rows,
        left=2,
    )

    # instruction mix of gemm as shares
    out += [
        "",
        "### GEMM instruction mix (share of the kernel's instructions, traced on the 2-layer sample)",
        "",
    ]
    rows = []
    for c in configs:
        r = kernels[c][GEMM]
        ratios = [number(r, f"{key}_per_work") for key, _ in MIX_CLASSES]
        total = None if None in ratios else sum(ratios)
        rows.append(
            [c]
            + [("-" if total is None else f"{100 * v / total:.1f}%") for v in ratios]
        )
    out += markdown_table(["config"] + [label for _, label in MIX_CLASSES], rows)

    # total over 50 layers
    out += ["", "### Total over the 50 weight layers (GEMM + requantization)", ""]
    rows, totals = [], {}
    for c in configs:
        gemm, requant = number(kernels[c][GEMM], "instret"), number(
            kernels[c][REQUANT], "instret"
        )
        totals[c] = gemm + requant
        rows.append(
            [
                c,
                count(gemm),
                count(requant),
                count(totals[c]),
                f"{totals[base] / totals[c]:.2f}x",
                f"{100 * requant / totals[c]:.1f}%",
                count(number(kernels[c][GEMM], "program_text_bytes")),
            ]
        )
    out += markdown_table(
        [
            "config",
            "GEMM instr",
            "requant instr",
            "total instr",
            "reduction vs " + base,
            "requant share",
            "program .text B",
        ],
        rows,
    )

    # per operator
    out += ["", "### Per operator (retired instructions, GEMM + requantization)", ""]
    rows = []
    for name, r in operators[base].items():
        row = [name, r["layers"], f"{100 * float(r['mac_share']):.1f}%"]
        row += [count(number(operators[c][name], "total_instret")) for c in configs]
        row += [
            f"{number(operators[base][name], 'total_instret') / number(operators[c][name], 'total_instret'):.2f}x"
            for c in others
        ]
        rows.append(row)
    out += markdown_table(
        ["operator", "layers", "MAC share"]
        + [f"instr {c}" for c in configs]
        + [f"reduction {c}" for c in others],
        rows,
    )

    # goals check
    out += [
        "",
        "### Check against other configuration - B0 (dynamic instructions for a major GEMM/FC kernel)",
        "",
    ]
    for c in others:
        gemm_reduction = number(kernels[base][GEMM], "instret") / number(
            kernels[c][GEMM], "instret"
        )
        per_operator = [
            number(operators[base][n], "total_instret")
            / number(operators[c][n], "total_instret")
            for n in operators[base]
        ]
        out.append(
            f"* {c}: GEMM kernel {gemm_reduction:.2f}x fewer instructions; every operator {min(per_operator):.2f}x "
            f"to {max(per_operator):.2f}x fewer (GEMM + requantization)"
        )
    out.append("")
    if not traced:
        out.append("")
        out.append(
            "WARNING: the instruction-mix columns are empty. Run profile_kernels.py without --no-mix for every config."
        )

    text = "\n".join(out) + "\n"
    print(text)
    path = os.path.join(RESULTS, "comparison.md")
    with open(path, "w") as f:
        f.write(text)
    print(f"wrote {os.path.normpath(path)}")


if __name__ == "__main__":
    main()
