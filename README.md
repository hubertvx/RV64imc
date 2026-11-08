# RV64IMC quantized-NN acceleration

This project measures how much a small custom RISC-V instruction reduces the instruction count of an
INT8-quantized transformer (DeiT-Tiny) running bare-metal on an RV64IMC core.

* **Workload:** the 50 weight layers of DeiT-Tiny:
  1,074,851,328 MACs = 85.7% of the model's 1,253,683,200 MACs. INT8 weights and activations (W8A8).
* **Custom instructions** (extension `Xqmac`): `qmac8 rd, rs1, rs2` multiplies 8 pairs of int8 values packed in two
  64-bit registers and adds the sum to `rd`. `qreq rd, rs1, rs2` turns one int32 accumulator into an int8 output
  (multiplier, rounding shift and clamp in one instruction). Both are plug-ins (extensions) for the Spike simulator.
* **Configurations:**

  | config | what it is | custom instruction |
  |:--|:--|:--|
  | **B0** | plain C on RV64IMC, the baseline | none |
  | **B1** | B0 with one software optimization: the K loop unrolled by 8 | none |
  | **X1** | B0's loops with `qmac8` in the inner loop | `qmac8` |
  | **X2** | X1 plus `qreq` for requantization | `qmac8`, `qreq` |
* **Metric:** retired instructions on Spike (ISA level)

## Results (reference only)

Produced on: **Ubuntu 26.04.1 LTS, x86_64** (Intel Core i5-1145G7)

| metric (50 weight layers)        |            B0 |            B1 |          X1 |          X2 |
|:---------------------------------|--------------:|--------------:|------------:|------------:|
| GEMM retired instructions        | 7,561,160,989 | 4,739,676,953 | 843,340,289 | 843,340,289 |
| GEMM instructions per MAC        |         7.035 |         4.410 |       0.785 |       0.785 |
| Requantization instructions      |   122,365,994 |   122,365,994 | 122,365,994 |  33,275,671 |
| Total (GEMM + requantization)    | 7,683,526,983 | 4,862,042,947 | 965,706,283 | 876,615,960 |
| Total vs B0                      |             - |   1.58x fewer |  7.96x fewer |  8.76x fewer |
| Requantization share of total    |          1.6% |          2.5% |       12.7% |        3.8% |
| GEMM kernel code size            |         226 B |         362 B |       220 B |       220 B |


Instruction mix of the GEMM kernel, per MAC (traced on a 2-layer sample of 50.8 M MACs; X2 is identical to X1 here):

| per MAC                    |    B0 |    B1 |    X1 |
|:---------------------------|------:|------:|------:|
| loads                      | 2.005 | 2.005 | 0.255 |
| stores                     | 0.005 | 0.005 | 0.005 |
| multiplies                 | 1.000 | 1.000 | 0.000 |
| branches / jumps           | 1.010 | 0.135 | 0.135 |
| other ALU (adds, pointers) | 3.026 | 1.276 | 0.276 |
| `qmac8`                    | 0.000 | 0.000 | 0.125 |

**What to take from it:** unrolling alone (B1) removes about a third of the instructions, because it only removes loop
overhead: the two loads and the multiply per MAC stay. `qmac8` removes the loads and multiplies as well, 5.62x fewer GEMM
instructions than B1. After X1, requantization is 12.7% of the total, and `qreq` brings it back to 3.8%. What remains is
the loop overhead inside the GEMM, and the operators that are not measured yet (see Limitations).

## Setup (Ubuntu)

```bash
# 1. system packages
sudo apt update
sudo apt install -y git make g++ rsync curl ca-certificates \
    device-tree-compiler libboost-dev libboost-regex-dev libboost-system-dev \
    gcc-riscv64-unknown-elf binutils-riscv64-unknown-elf
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. get the project
git clone https://github.com/hubertvx/RV64imc.git
cd RV64imc

# 3. Python environment, model, quantized weights
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
.venv/bin/python scripts/get_model.py                    # downloads models/model.tflite
.venv/bin/python scripts/get_calib_images.py             # downloads 1,256 COCO images 
.venv/bin/python scripts/quantize_w8a8.py                # writes models/deit_tiny_int8.npz

# 4. Spike, at the exact version this project was built with
git clone https://github.com/riscv-software-src/riscv-isa-sim.git src/isa/riscv-isa-sim
git -C src/isa/riscv-isa-sim checkout 5a760667bf0d0ab893173b9eeff9c109c8d7dd9a
mkdir src/isa/riscv-isa-sim/build
REPO=$PWD
cd src/isa/riscv-isa-sim/build && ../configure --prefix="$REPO/install" && make -j"$(nproc)" && make install

cd ../../../.. # back to main repo if currently at src/isa/riscv-isa-sim/build
```


Step 4 must use that exact Spike commit: different versions might have a different layout and the build fails.
If already have the CV-Wally toolchain ([cvw toolchain installation](https://github.com/openhwfoundation/cvw#toolchain-installation-and-configuration)), put its tools on `PATH` (or set `RISCV=/opt/riscv`); the Makefile also
searches there.


## Setup (macOS)
Tested on Apple Silicon (arm64).
```bash
# 1. command line tools, then Homebrew packages
xcode-select --install

brew install uv

brew trust riscv-software-src/riscv
brew tap riscv-software-src/riscv
brew install riscv-tools

# 2. get the project, Python environment, model: same as Ubuntu steps 2 and 3

# 3. Spike: same as Ubuntu step 4

```

## Calibration and evaluation images

Quantizing to int8 needs real images to find the range of every layer's input (the "scale"). The project uses a fixed
subset of **COCO val2017**, listed in `data/coco_manifest.csv` with the sha256 hash of every file. The images are
downloaded by `scripts/get_calib_images.py` and are **not stored in the repository** (`data/coco/` is git-ignored).

| set | images | used for |
|:--|--:|:--|
| `data/coco/calib` | 256 | the only images that set the activation scales (`quantize_w8a8.py`) |
| `data/coco/eval` | 1,000 | never used for calibration: the per-layer error check (first 32), the test image of the benchmark, and the accuracy measurement of the whole model later |

* Chosen with a fixed seed (2026) from the 1,279 val2017 images under CC BY, CC BY-SA or "no known copyright
  restrictions" licences. The licence and the Flickr address of every image are in the manifest.
* The instruction counts do not depend on the images, only on the layer shapes.

## Run

```bash
make check                                  # builds everything missing, runs every test
python3 scripts/profile_kernels.py --config b0      
python3 scripts/profile_kernels.py --config b1      
python3 scripts/profile_kernels.py --config x1      
python3 scripts/profile_kernels.py --config x2      
python3 scripts/compare_configs.py b0 b1 x1 x2      
```



`make check` generates the test data (`scripts/export_c.py` runs automatically), builds the Spike extension, and runs:
the runtime smoke test, the kernel unit tests (hand-checked values, and random shapes including odd sizes for B0 and B1),
the `qmac8` test (534,309 checks) and the `qreq` test (5,480,059 checks), B0, B1, X1 and X2 on all 50 layers against
numpy, and a check that the programs contain no floating-point instruction.



### The Spike commands behind it

Do not have to type any Spike command. They are written inside the project, so `make` and the Python script run
them. The table shows a few examples of these commands and where each one is defined:

| Spike command | Defined in |
|:--|:--|
| run B0: `spike --isa=rv64imc_zicsr build/bench_layer_b0.elf` | `Makefile` (`run`, `test`); `scripts/profile_kernels.py` (`run_program()`) |
| run X1: `install/bin/spike --extlib=build/libqmac8.so --extension=qmac8 --isa=rv64imc_zicsr build/bench_layer_x1.elf` | the same two files (`QEXT` in the Makefile, `main()` in the script) |
| trace every instruction: the same command plus `-l` | `scripts/profile_kernels.py` (`trace_mix()`; `tools/instr_mix.c` reads the log) |
| run X2: `install/bin/spike --extlib=build/libqreq.so --extension=qreq --isa=rv64imc_zicsr build/bench_layer_x2.elf` | `Makefile` (`QEXT2`); `scripts/profile_kernels.py` |
| test the new instructions: the X1 / X2 command running `build/test_isa_qmac8.elf` and `build/test_isa_qreq.elf` | `Makefile` (`qtest`) |
| test the runtime: `spike --isa=rv64imc_zicsr build/smoke_runtime.elf` | `Makefile` (`smoke`) |


Some flags used and its meaning

| flag | meaning |
|:--|:--|
| `--isa=rv64imc_zicsr` | which RISC-V core Spike simulates (RV64IMC; `zicsr` lets the programs read the `minstret` counter with `csrr`) |
| `--extlib=build/libqmac8.so` | load a compiled extension library (it contains the definition of `qmac8`) |
| `--extension=qmac8` | switch on the extension of that name (`qreq` for `libqreq.so`) |
| `-l` | print every executed instruction (address, encoding, name) |





## Repository Structure

```
scripts/       get_model.py, get_calib_images.py, quantize_w8a8.py, export_c.py        model -> C data
               profile_kernels.py, compare_configs.py                                  measurement and comparison
               analyze_base_model.py, crosscheck_macs.py                               MAC counts of the model
src/runtime/   startup.S, linker.ld, runtime.c/.h, model_data.S                        bare-metal runtime for Spike
src/kernels/   kernels.h, gemm_b0.c, gemm_b1.c, gemm_x1.c, requant.c, requant_x2.c,
               qmac8.h, qreq.h                                                         the code that is measured
src/bench/     bench_layer.c                                                           runs and checks every layer
src/isa/qmac8/ qmac8.cc, build_ext.mk                                                  qmac8 and qreq for Spike
tests/         smoke_runtime.c, test_kernels_host.c, test_gemm_random_host.c,
               test_isa_qmac8.c, test_isa_qreq.c
tools/         instr_mix.c                                                             classifies Spike's instruction trace
results/       b0*.csv, b1*.csv, x1*.csv, x2*.csv, comparison.md, deit_tiny_macs.csv   reference measurements
```


## Limitations

* Retired instruction counts on Spike, **not cycles**. Cycles, CPI, clock and area need the CV-Wally RTL.
* The measured workload is the 50 layers with stored weights (~86% of the MACs).
* Only the 50 weight layers are measured. The attention matmuls, softmax, GELU, LayerNorm and residual adds are not;
* B1 applies a single standard optimization (unrolling). A blocked or hand-scheduled kernel could be faster, so
  the X1-vs-B1 ratio is an upper estimate of the gain over good C code.


## Credits

DeiT-Tiny from `litert-community/deit_tiny_patch16_224` (LiteRT model on Hugging Face)
