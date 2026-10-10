#!/usr/bin/env python3
"""
Profile one kernel configuration on Spike and write results

  Counts: the benchmark runs ALL 50 weight layers of deit-tiny (about 86% of the model's MACs). Retired instructions come from the counter the program reads itself (exact), per layer and per operator.
  Mix: tracing every instruction (spike -l) is about 500x slower than a plain run, so the load / store, multiply, branch, custom, ALU mix is traced on a 2-layer sample. Every layer runs the same kernel loop, so the mix per MAC is the same for all layers. The sample's counters are checked to equal the full run's counters for those layers
  Code size: kernel functions and the whole .text of the program

The program checks every layer against numpy (byte for byte): a wrong kernel fails here (exit code != 0)
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


def run_program(spike, isa, elf):
    """Run the benchmark on Spike; return one dict per layer ('== name' section of the output)."""
    res = run([spike, f"--isa={isa}", elf], capture_output=True)
    vecs = []
    for line in res.stdout.splitlines():
        if line.startswith("== "):
            vecs.append({"name": line[3:]})
        elif ": " in line and vecs:
            k, v = line.split(": ", 1)
            if v.strip().isdigit():
                vecs[-1][k] = int(v)
    verdict = res.stdout.strip().splitlines()[-1]
    if res.returncode != 0:
        sys.exit(f"{elf}: the program's own check failed ({verdict}): kernel output does not match numpy")
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


def text_bytes(elf):
    """Size of all code sections (.text*) of the whole program."""
    out = run([PREFIX + "size", "-A", elf], capture_output=True, check=True).stdout
    return sum(int(l.split()[1]) for l in out.splitlines() if l.startswith(".text"))


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


def operator_table(vecs):
    """One row per operator kind (blocks_0_mlp_fc1 ... blocks_11_mlp_fc1 -> mlp_fc1)."""
    ops = {}
    for v in vecs:
        o = ops.setdefault(re.sub(r"blocks_\d+_", "", v["name"]),
                           {"layers": 0, "macs": 0, "gemm": 0, "requant": 0})
        o["layers"] += 1
        o["macs"] += v["MACS"]
        o["gemm"] += v["INSTRET_GEMM"]
        o["requant"] += v["INSTRET_REQUANT"]
    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="b0")
    ap.add_argument("--no-mix", action="store_true", help="skip the slow instruction trace")
    ap.add_argument("--spike", default="/opt/riscv/bin/spike")
    ap.add_argument("--isa", default="rv64imc_zicsr")
    args = ap.parse_args()
    cfg = args.config

    spike = args.spike

    elf = f"build/bench_layer_{cfg}.elf"
    run(["make", "-s", f"CONFIG={cfg}", elf], check=True)
    vecs = run_program(spike, args.isa, elf)
    print(f"PASS: {len(vecs)} layers match numpy byte for byte")

    macs = sum(v["MACS"] for v in vecs)
    outputs = sum(v["M"] * v["N"] for v in vecs)
    instret = {"gemm_s8": sum(v["INSTRET_GEMM"] for v in vecs), "requant": sum(v["INSTRET_REQUANT"] for v in vecs)}
    work = {"gemm_s8": macs, "requant": outputs}
    print(f"exact counts: gemm_s8 {instret['gemm_s8']:,} instructions for {macs:,} MACs "
          f"({instret['gemm_s8'] / macs:.3f} per MAC); requant {instret['requant']:,} instructions "
          f"for {outputs:,} outputs", flush=True)

    mix = None
    if not args.no_mix:
        print("tracing the instruction mix on the 2-layer sample with spike -l (about 8 minutes) ...", flush=True)
        sample_elf = f"build/bench_layer_{cfg}_sample.elf"
        run(["make", "-s", f"CONFIG={cfg}", "SAMPLE=1", sample_elf], check=True)
        svecs = run_program(spike, args.isa, sample_elf)
        for sv, fv in zip(svecs, vecs):
            assert sv["name"] == fv["name"], f"layer order differs: {sv['name']} vs {fv['name']}"
            for key in sv:
                if key == "name":
                    continue
                tol = 8 if key.startswith("INSTRET") else 0
                assert abs(sv[key] - fv[key]) <= tol, \
                    f"sample run differs from full run for {sv['name']}: {key} {sv[key]} vs {fv[key]}"
        sranges, _ = symbol_ranges(os.path.join(ROOT, sample_elf))
        mix = trace_mix(spike, args.isa, os.path.join(ROOT, sample_elf), sranges)
        swork = {"gemm_s8": sum(v["MACS"] for v in svecs), "requant": sum(v["M"] * v["N"] for v in svecs)}

    _, size = symbol_ranges(os.path.join(ROOT, elf))
    elf_text = text_bytes(os.path.join(ROOT, elf))

    cols = ["config", "kernel", "instret", "work", "instr_per_work", "mac_per_instr", "load_per_work",
            "store_per_work", "mul_per_work", "branch_per_work", "custom_per_work", "alu_per_work",
            "ldst_per_work", "kernel_code_bytes", "program_text_bytes", "mix_sample_work"]
    rows = []
    for k in KERNELS:
        r = {"config": cfg, "kernel": k, "instret": instret[k], "work": work[k],
             "instr_per_work": f"{instret[k] / work[k]:.3f}",
             "mac_per_instr": f"{work[k] / instret[k]:.4f}" if k == "gemm_s8" else "",
             "kernel_code_bytes": size[k], "program_text_bytes": elf_text}
        if mix:
            _, ld, st, mu, br, cu, al = mix[k]
            w = swork[k]
            r.update(load_per_work=f"{ld / w:.4f}", store_per_work=f"{st / w:.4f}", mul_per_work=f"{mu / w:.4f}",
                     branch_per_work=f"{br / w:.4f}", custom_per_work=f"{cu / w:.4f}", alu_per_work=f"{al / w:.4f}",
                     ldst_per_work=f"{(ld + st) / w:.4f}", mix_sample_work=w)
        rows.append(r)

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    path = os.path.join(ROOT, "results", f"{cfg}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    ops = operator_table(vecs)
    total_instr = sum(o["gemm"] + o["requant"] for o in ops.values())
    op_path = os.path.join(ROOT, "results", f"{cfg}_operators.csv")
    with open(op_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "operator", "layers", "macs", "mac_share", "gemm_instret", "requant_instret",
                    "total_instret", "instret_share", "instr_per_mac"])
        for name, o in ops.items():
            t = o["gemm"] + o["requant"]
            w.writerow([cfg, name, o["layers"], o["macs"], f"{o['macs'] / macs:.4f}", o["gemm"], o["requant"], t,
                        f"{t / total_instr:.4f}", f"{t / o['macs']:.3f}"])

    print(f"\nconfig {cfg}: {len(vecs)} layers, {macs:,} MACs ({100 * macs / 1253683200:.1f}% of the model), "
          f"{outputs:,} outputs")
    print(f"program code (.text): {elf_text:,} bytes")

    print(f"\n{'operator':<18}{'layers':>7}{'MACs':>16}{'MAC %':>7}{'instructions':>17}{'instr %':>9}{'instr/MAC':>11}")
    for name, o in ops.items():
        t = o["gemm"] + o["requant"]
        print(f"{name:<18}{o['layers']:>7}{o['macs']:>16,}{100 * o['macs'] / macs:>6.1f}%{t:>17,}"
              f"{100 * t / total_instr:>8.1f}%{t / o['macs']:>11.3f}")
    print(f"{'TOTAL':<18}{len(vecs):>7}{macs:>16,}{'':>7}{total_instr:>17,}")

    for r in rows:
        k = r["kernel"]
        print(f"\n{k}  (work = {'MACs' if k == 'gemm_s8' else 'outputs'}, kernel code {r['kernel_code_bytes']} bytes)")
        print(f"  retired instructions : {r['instret']:,}   ({r['instr_per_work']} per work item)")
        if r["mac_per_instr"]:
            print(f"  MACs per instruction : {r['mac_per_instr']}")
        if mix:
            t = mix[k][0]
            print(f"  mix from the {r['mix_sample_work']:,}-work-item sample ({t:,} traced instructions):")
            for label, key, c in (("load", "load", 1), ("store", "store", 2), ("multiply", "mul", 3),
                                  ("branch/jump", "branch", 4), ("custom", "custom", 5), ("other ALU", "alu", 6)):
                print(f"    {label:<12}{mix[k][c]:>14,}  {100 * mix[k][c] / t:5.1f}%   {r[key + '_per_work']} per work item")
    print(f"\nwrote {os.path.normpath(path)} and {os.path.normpath(op_path)}")


if __name__ == "__main__":
    main()
