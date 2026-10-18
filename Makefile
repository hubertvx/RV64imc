# RV64IMC quantized-NN acceleration (DeiT-Tiny on Spike)

#   make gen                 export the quantized model (needs the Python venv)
#   make CONFIG=b0 run       build and run the layer benchmark for one kernel configuration
#   make CONFIG=b0 test      same, but only print PASS/FAIL (exit code 0 = outputs match numpy)
#   make nofp                check that the ELF contains no floating-point instructions
#   make qext                build the qmac8 Spike extension (build/libqmac8.so)
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

.PHONY: all gen run test smoke ktest qext qtest nofp clean
all: $(ELF)

gen: $(GEN)/weights.bin
$(GEN)/weights.bin $(GEN)/vectors.bin $(GEN)/model.h:
	.venv/bin/python scripts/export_c.py

$(ELF): $(SRCS) src/kernels/kernels.h $(GEN)/weights.bin
	@mkdir -p build
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(SRCS)

run: $(ELF)
	$(SPIKE) --isa=$(ISA) $(ELF)

test: $(ELF)
	@$(SPIKE) --isa=$(ISA) $(ELF) > build/test.log; r=$$?; tail -1 build/test.log; exit $$r

ktest:
	@mkdir -p build
	gcc -O2 -Wall -Wextra -Isrc/kernels -o build/ktest tests/test_kernels_host.c src/kernels/gemm_b0.c src/kernels/requant.c
	build/ktest

build/libqmac8.so: src/isa/qmac8/qmac8.cc src/isa/qmac8/build_ext.mk
	$(MAKE) -f src/isa/qmac8/build_ext.mk

qext: build/libqmac8.so

# qmac8 is an illegal instruction on Spike
# --foreground: without it, timeout started by make's non-interactive shell runs in a background process group
# and the terminal STOPS it after the test done, so make qtest never returns.
QTEST := build/test_isa_qmac8.elf
qtest: qext
	@mkdir -p build
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(QTEST) src/runtime/startup.S src/runtime/runtime.c tests/test_isa_qmac8.c
	timeout --foreground 20 $(SPIKE_X) $(QEXT) --isa=$(ISA) $(QTEST)

SMOKE := build/smoke_runtime.elf
smoke:
	@mkdir -p build
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(SMOKE) src/runtime/startup.S src/runtime/runtime.c tests/smoke_runtime.c
	$(SPIKE) --isa=$(ISA) $(SMOKE)

nofp: $(ELF)
	@if $(OBJDUMP) -d $(ELF) | grep -E '\s(f(add|sub|mul|div|sqrt|lw|ld|sw|sd|cvt|mv|eq|lt|le|madd)\.|c\.f)'; then \
	    echo "FAIL: floating-point instruction found"; exit 1; \
	else echo "OK: no floating-point instructions in $(ELF)"; fi

clean:
	rm -f build/*.elf
