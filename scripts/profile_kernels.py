#!/usr/bin/env python3
"""
Measure one kernel configuration on Spike and write results/<config>.csv.

Per kernel (gemm_s8, requant) it reports:
  * retired instructions            (counter read by the program)
  * instruction mix                 (load / store / multiply / branch+jump / custom / other ALU),
                                    from the Spike instruction trace, restricted to the kernel's code
  * work: MACs for the GEMM, output elements for requant, and the per-work ratios
  * code size of the kernel in bytes

The program also checks its own results against numpy, so a wrong kernel fails here (exit code != 0).
"""
import argparse
import csv
import os
import re
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PREFIX = "/opt/riscv/bin/riscv64-unknown-elf-"
KERNELS = {                    # name -> symbols whose code belongs to it
    "gemm_s8": ["gemm_s8"],
    "requant": ["requant", "high_mul", "round_shift"],
}


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, text=True, **kw)


def parse_program_output(text):
    """Sections '== name' followed by 'KEY: value' lines -> list of dicts."""
    vecs = []
    for line in text.splitlines():
        if line.startswith("== "):
            vecs.append({"name": line[3:]})
        elif ": " in line and vecs:
            k, v = line.split(": ", 1)
            if v.strip().isdigit():
                vecs[-1][k] = int(v)
    return vecs


def symbol_ranges(elf):
    out = run([PREFIX + "nm", "-S", "--defined-only", elf], capture_output=True, check=True).stdout
    syms = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[2] in ("T", "t"):
            syms[parts[3]] = (int(parts[0], 16), int(parts[1], 16))
    ranges, size = {}, {}
    for kernel, names in KERNELS.items():
        rs = [(syms[n][0], syms[n][0] + syms[n][1]) for n in names if n in syms]
        if not rs:
            raise SystemExit(f"symbol(s) {names} not found in {elf}")
        ranges[kernel] = rs
        size[kernel] = sum(hi - lo for lo, hi in rs)
    return ranges, size


def trace_mix(spike, isa, elf, ranges):
    exe = os.path.join(ROOT, "build", "mixcount")
    run(["gcc", "-O2", "-o", exe, "tools/instr_mix.c"], check=True)
    args = [f"{k}@{i}={lo:x}-{hi:x}" for k, rs in ranges.items() for i, (lo, hi) in enumerate(rs)]
    sp = subprocess.Popen([spike, "-l", f"--isa={isa}", elf], cwd=ROOT, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE)
    mc = subprocess.run([exe] + args, stdin=sp.stderr, capture_output=True, text=True, check=True)
    sp.wait()
    mix = {k: [0] * 7 for k in ranges}
    for line in mc.stdout.splitlines():
        name, *vals = line.split()
        kernel = re.sub(r"@\d+$", "", name)
        mix[kernel] = [a + int(b) for a, b in zip(mix[kernel], vals)]
    return mix          # kernel -> [total, load, store, mul, branch, custom, alu]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="b0")
    ap.add_argument("--no-mix", action="store_true", help="skip the slow instruction trace")
    ap.add_argument("--spike", default="/opt/riscv/bin/spike")
    ap.add_argument("--isa", default="rv64imc_zicsr")
    args = ap.parse_args()

    elf = f"build/bench_layer_{args.config}.elf"
    run(["make", "-s", f"CONFIG={args.config}", elf], check=True)

    res = run([args.spike, f"--isa={args.isa}", elf], capture_output=True)
    vecs = parse_program_output(res.stdout)
    print(res.stdout.strip().splitlines()[-1], f"(exit code {res.returncode})")
    if res.returncode != 0:
        sys.exit("the program's own check failed: kernel output does not match numpy")

    macs = sum(v["MACS"] for v in vecs)
    outputs = sum(v["M"] * v["N"] for v in vecs)
    instret = {"gemm_s8": sum(v["INSTRET_GEMM"] for v in vecs), "requant": sum(v["INSTRET_REQUANT"] for v in vecs)}
    work = {"gemm_s8": macs, "requant": outputs}

    print(f"exact counts: gemm_s8 {instret['gemm_s8']:,} instructions for {macs:,} MACs "
          f"({instret['gemm_s8'] / macs:.3f} per MAC); requant {instret['requant']:,} instructions "
          f"for {outputs:,} outputs", flush=True)
    if not args.no_mix:
        print("tracing the instruction mix with spike -l (takes about 8 minutes) ...", flush=True)

    ranges, size = symbol_ranges(os.path.join(ROOT, elf))
    mix = None if args.no_mix else trace_mix(args.spike, args.isa, os.path.join(ROOT, elf), ranges)

    cols = ["config", "kernel", "instret", "work", "instr_per_work", "mac_per_instr", "load", "store",
            "mul", "branch", "custom", "alu", "ldst_per_work", "code_bytes"]
    rows = []
    for k in KERNELS:
        r = {"config": args.config, "kernel": k, "instret": instret[k], "work": work[k],
             "instr_per_work": f"{instret[k] / work[k]:.3f}",
             "mac_per_instr": f"{work[k] / instret[k]:.4f}" if k == "gemm_s8" else "",
             "code_bytes": size[k]}
        if mix:
            _, ld, st, mu, br, cu, al = mix[k]
            r.update(load=ld, store=st, mul=mu, branch=br, custom=cu, alu=al,
                     ldst_per_work=f"{(ld + st) / work[k]:.3f}")
        rows.append(r)

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    path = os.path.join(ROOT, "results", f"{args.config}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"\nconfig {args.config}: {len(vecs)} layers, {macs:,} MACs, {outputs:,} outputs")
    for r in rows:
        print(f"\n{r['kernel']}  (work = {'MACs' if r['kernel'] == 'gemm_s8' else 'outputs'}, code {r['code_bytes']} bytes)")
        print(f"  retired instructions : {r['instret']:,}   ({r['instr_per_work']} per work item)")
        if r["mac_per_instr"]:
            print(f"  MACs per instruction : {r['mac_per_instr']}")
        if mix:
            t = mix[r["kernel"]][0]
            print(f"  traced instructions  : {t:,}  (counter: {r['instret']:,})")
            for label, key in (("load", "load"), ("store", "store"), ("multiply", "mul"),
                               ("branch/jump", "branch"), ("custom", "custom"), ("other ALU", "alu")):
                v = r[key]
                print(f"  {label:<12}{v:>14,}  {100 * v / t:5.1f}%   {v / work[r['kernel']]:.3f} per work item")
    print(f"\nwrote {os.path.normpath(path)}")


if __name__ == "__main__":
    main()
