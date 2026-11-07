# RV64IMC quantized-NN acceleration (DeiT-Tiny on Spike)

#   make gen                 export the quantized model (needs the Python venv)
#   make CONFIG=b0 run       build and run the layer benchmark for one kernel configuration (b0, b1, x1 or x2)
#   make CONFIG=b0 test      same, but only print PASS/FAIL (exit code 0 = outputs match numpy)
#   make nofp                check that the ELF contains no floating-point instructions
#   make check               run every test: runtime, kernels, qmac8, qreq, B0/B1/X1/X2 vs numpy, no FP
#   make bench               full evaluation: profile B0, B1, X1 and X2 with instruction mix, then compare
#   make qext                build the Spike extensions: libqmac8.so (qmac8) and libqreq.so (qmac8 + qreq)
#   make qtest               run tests/test_isa_qmac8.c (534,309 checks) and tests/test_isa_qreq.c on Spike
#   make smoke               build and run tests/smoke_runtime.c (runtime only, no model data)
#   make ktest               compile and run the kernel unit test on the PC (tests/test_kernels_host.c)

RISCV   ?= /opt/riscv
# The RISC-V compiler and binutils: in $(RISCV)/bin if a toolchain is installed there (the CV-Wally layout),
# otherwise whatever is on PATH (apt, Homebrew, one you built yourself). A different name prefix can be given,
# e.g.  make PREFIX=riscv64-elf-
TOOLDIR := $(if $(wildcard $(RISCV)/bin/riscv64-unknown-elf-gcc),$(RISCV)/bin/,)
PREFIX  ?= $(TOOLDIR)riscv64-unknown-elf-
CC       = $(PREFIX)gcc
OBJDUMP  = $(PREFIX)objdump
# Spike without the extension (used for b0): $(RISCV)/bin/spike if present, else our own build, else spike on PATH
SPIKE   ?= $(firstword $(wildcard $(RISCV)/bin/spike install/bin/spike) spike)
# GNU timeout: "timeout" on Linux, "gtimeout" on macOS (brew install coreutils); without it the tests run unguarded
TIMEOUT := $(shell command -v timeout || command -v gtimeout)
# Spike with the custom instruction: our own build plus the qmac8 extension library
SPIKE_X ?= install/bin/spike
QEXT    := --extlib=build/libqmac8.so --extension=qmac8
# X2 uses both custom instructions: libqreq.so is the same source built with -DWITH_QREQ (extension name qreq)
QEXT2   := --extlib=build/libqreq.so --extension=qreq

ISA     ?= rv64imc_zicsr
CONFIG  ?= b0

# b0 and b1 are plain RV64IMC and run on stock Spike; x1 needs qmac8, x2 needs qmac8 and qreq
GEMM_SRC := src/kernels/gemm_$(CONFIG).c
REQ_SRC  := src/kernels/requant.c
ifneq ($(filter $(CONFIG),b0 b1),)
RUNSPIKE := $(SPIKE)
RUNDEPS  :=
else ifeq ($(CONFIG),x2)
RUNSPIKE := $(SPIKE_X) $(QEXT2)
RUNDEPS  := build/libqreq.so
GEMM_SRC := src/kernels/gemm_x1.c
REQ_SRC  := src/kernels/requant_x2.c
else
RUNSPIKE := $(SPIKE_X) $(QEXT)
RUNDEPS  := build/libqmac8.so
endif
GEN     := build/gen
# SAMPLE=1 builds the variant that runs only the first NUM_SAMPLE_VECTORS layers (for instruction tracing)
ifeq ($(SAMPLE),1)
SUFFIX  := _sample
EXTRA_CFLAGS += -DSAMPLE_ONLY
endif
ELF     := build/bench_layer_$(CONFIG)$(SUFFIX).elf

CFLAGS  := -O2 -march=$(ISA) -mabi=lp64 -mcmodel=medany -ffreestanding -fno-builtin \
           -fno-tree-loop-distribute-patterns -nostdlib -nostartfiles -Wall -Wextra \
           -Isrc/runtime -I$(GEN) $(EXTRA_CFLAGS)
LDFLAGS := -T src/runtime/linker.ld -Wl,--no-warn-rwx-segments

SRCS    := src/runtime/startup.S src/runtime/runtime.c src/runtime/model_data.S \
           src/bench/bench_layer.c $(GEMM_SRC) $(REQ_SRC)

.PHONY: all gen run test smoke ktest qext qtest nofp clean check bench
all: $(ELF)

gen: $(GEN)/weights.bin
$(GEN)/weights.bin $(GEN)/vectors.bin $(GEN)/model.h:
	.venv/bin/python scripts/export_c.py

$(ELF): $(SRCS) $(wildcard src/kernels/*.h) $(GEN)/weights.bin
	@mkdir -p build
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(SRCS)

run: $(ELF) $(RUNDEPS)
	$(RUNSPIKE) --isa=$(ISA) $(ELF)

test: $(ELF) $(RUNDEPS)
	@$(RUNSPIKE) --isa=$(ISA) $(ELF) > build/test.log; r=$$?; tail -1 build/test.log; exit $$r

ktest:
	@mkdir -p build
	gcc -O2 -Wall -Wextra -Isrc/kernels -o build/ktest tests/test_kernels_host.c src/kernels/gemm_b0.c src/kernels/requant.c
	build/ktest
	@# the hand-checked kernel test again with the B1 kernel, and random shapes (odd M and N too) for both
	gcc -O2 -Wall -Wextra -Isrc/kernels -o build/ktest_b1 tests/test_kernels_host.c src/kernels/gemm_b1.c src/kernels/requant.c
	build/ktest_b1
	gcc -O2 -Wall -Wextra -Isrc/kernels -o build/ktest_rand_b0 tests/test_gemm_random_host.c src/kernels/gemm_b0.c
	build/ktest_rand_b0
	gcc -O2 -Wall -Wextra -Isrc/kernels -o build/ktest_rand_b1 tests/test_gemm_random_host.c src/kernels/gemm_b1.c
	build/ktest_rand_b1

build/libqmac8.so: src/isa/qmac8/qmac8.cc src/isa/qmac8/build_ext.mk
	$(MAKE) -f src/isa/qmac8/build_ext.mk

# the same source with -DWITH_QREQ: adds the qreq instruction (the extension is then called qreq)
build/libqreq.so: src/isa/qmac8/qmac8.cc src/isa/qmac8/build_ext.mk
	$(MAKE) -f src/isa/qmac8/build_ext.mk OUT=build/libqreq.so EXTRA=-DWITH_QREQ

qext: build/libqmac8.so build/libqreq.so

QTEST := build/test_isa_qmac8.elf
QTEST2 := build/test_isa_qreq.elf
qtest: qext
	@mkdir -p build
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(QTEST) src/runtime/startup.S src/runtime/runtime.c tests/test_isa_qmac8.c
	$(if $(TIMEOUT),$(TIMEOUT) --foreground 60) $(SPIKE_X) $(QEXT) --isa=$(ISA) $(QTEST)
	@# qmac8 must work unchanged in the library that also has qreq (X2 uses both)
	$(if $(TIMEOUT),$(TIMEOUT) --foreground 60) $(SPIKE_X) $(QEXT2) --isa=$(ISA) $(QTEST)
	$(CC) $(CFLAGS) -Isrc/kernels $(LDFLAGS) -o $(QTEST2) src/runtime/startup.S src/runtime/runtime.c tests/test_isa_qreq.c
	$(if $(TIMEOUT),$(TIMEOUT) --foreground 120) $(SPIKE_X) $(QEXT2) --isa=$(ISA) $(QTEST2)

SMOKE := build/smoke_runtime.elf
smoke:
	@mkdir -p build
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(SMOKE) src/runtime/startup.S src/runtime/runtime.c tests/smoke_runtime.c
	$(SPIKE) --isa=$(ISA) $(SMOKE)

nofp: $(ELF)
	@if $(OBJDUMP) -d $(ELF) | grep -E '[[:space:]](f(add|sub|mul|div|sqrt|lw|ld|sw|sd|cvt|mv|eq|lt|le|madd)\.|c\.f)'; then \
	    echo "FAIL: floating-point instruction found"; exit 1; \
	else echo "OK: no floating-point instructions in $(ELF)"; fi

# every test; stops at the first failure
check: smoke ktest qtest
	$(MAKE) --no-print-directory CONFIG=b0 test
	$(MAKE) --no-print-directory CONFIG=b1 test
	$(MAKE) --no-print-directory CONFIG=x1 test
	$(MAKE) --no-print-directory CONFIG=x2 test
	$(MAKE) --no-print-directory CONFIG=b0 nofp
	$(MAKE) --no-print-directory CONFIG=b1 nofp
	$(MAKE) --no-print-directory CONFIG=x1 nofp
	$(MAKE) --no-print-directory CONFIG=x2 nofp
	@echo "ALL CHECKS PASSED"

# the full evaluation: writes results/<config>*.csv for b0, b1, x1, x2 and results/comparison.md
bench: gen qext
	python3 scripts/profile_kernels.py --config b0
	python3 scripts/profile_kernels.py --config b1
	python3 scripts/profile_kernels.py --config x1
	python3 scripts/profile_kernels.py --config x2
	python3 scripts/compare_configs.py b0 b1 x1 x2

clean:
	rm -f build/*.elf
