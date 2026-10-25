# RV64IMC quantized-NN acceleration (DeiT-Tiny on Spike)

#   make gen                 export the quantized model (needs the Python venv)
#   make CONFIG=b0 run       build and run the layer benchmark for one kernel configuration (b0 or x1)
#   make CONFIG=b0 test      same, but only print PASS/FAIL (exit code 0 = outputs match numpy)
#   make nofp                check that the ELF contains no floating-point instructions
#   make check               run every test: runtime, kernels, qmac8, B0 and X1 vs numpy, no FP
#   make bench               full evaluation: profile B0 and X1 with instruction mix, then compare
#   make qext                build the qmac8 Spike extension
#   make qtest               run tests/test_isa_qmac8.c on Spike with the qmac8 extension (534,309 checks)
#   make smoke               build and run tests/smoke_runtime.c (runtime only, no model data)
#   make ktest               compile and run the kernel unit test on the PC (tests/test_kernels_host.c)

RISCV   ?= /opt/riscv
PREFIX  ?= $(RISCV)/bin/riscv64-unknown-elf-
CC       = $(PREFIX)gcc
OBJDUMP  = $(PREFIX)objdump
SPIKE   ?= $(RISCV)/bin/spike
# Spike with the custom instruction: our own build plus the qmac8 extension library
SPIKE_X ?= install/bin/spike
QEXT    := --extlib=build/libqmac8.so --extension=qmac8

ISA     ?= rv64imc_zicsr
CONFIG  ?= b0

# b0 runs on stock Spike; every other configuration uses the custom instruction and needs the extension loaded
ifeq ($(CONFIG),b0)
RUNSPIKE := $(SPIKE)
RUNDEPS  :=
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
           src/bench/bench_layer.c src/kernels/gemm_$(CONFIG).c src/kernels/requant.c

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

build/libqmac8.so: src/isa/qmac8/qmac8.cc src/isa/qmac8/build_ext.mk
	$(MAKE) -f src/isa/qmac8/build_ext.mk

qext: build/libqmac8.so

QTEST := build/test_isa_qmac8.elf
qtest: qext
	@mkdir -p build
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(QTEST) src/runtime/startup.S src/runtime/runtime.c tests/test_isa_qmac8.c
	timeout --foreground 60 $(SPIKE_X) $(QEXT) --isa=$(ISA) $(QTEST)

SMOKE := build/smoke_runtime.elf
smoke:
	@mkdir -p build
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(SMOKE) src/runtime/startup.S src/runtime/runtime.c tests/smoke_runtime.c
	$(SPIKE) --isa=$(ISA) $(SMOKE)

nofp: $(ELF)
	@if $(OBJDUMP) -d $(ELF) | grep -E '\s(f(add|sub|mul|div|sqrt|lw|ld|sw|sd|cvt|mv|eq|lt|le|madd)\.|c\.f)'; then \
	    echo "FAIL: floating-point instruction found"; exit 1; \
	else echo "OK: no floating-point instructions in $(ELF)"; fi

# every test; stops at the first failure
check: smoke ktest qtest
	$(MAKE) --no-print-directory CONFIG=b0 test
	$(MAKE) --no-print-directory CONFIG=x1 test
	$(MAKE) --no-print-directory CONFIG=b0 nofp
	$(MAKE) --no-print-directory CONFIG=x1 nofp
	@echo "ALL CHECKS PASSED"

# the full evaluation: writes results/b0*.csv, results/x1*.csv and results/comparison.md
bench: gen qext
	python3 scripts/profile_kernels.py --config b0
	python3 scripts/profile_kernels.py --config x1
	python3 scripts/compare_configs.py b0 x1

clean:
	rm -f build/*.elf
