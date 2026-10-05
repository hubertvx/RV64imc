# RV64IMC quantized-NN acceleration (DeiT-Tiny on Spike)

#   make gen                 export the quantized model (needs the Python venv)
#   make CONFIG=b0 run       build and run the layer benchmark for one kernel configuration
#   make CONFIG=b0 test      same, but only print PASS/FAIL (exit code 0 = outputs match numpy)
#   make nofp                check that the ELF contains no floating-point instructions
#   make smoke               build and run tests/smoke_runtime.c (runtime only, no model data)
#   make ktest               compile and run the kernel unit test on the PC (tests/test_kernels_host.c)

RISCV   ?= /opt/riscv
PREFIX  ?= $(RISCV)/bin/riscv64-unknown-elf-
CC       = $(PREFIX)gcc
OBJDUMP  = $(PREFIX)objdump
SPIKE   ?= $(RISCV)/bin/spike

ISA     ?= rv64imc_zicsr
CONFIG  ?= b0
GEN     := build/gen
ELF     := build/bench_layer_$(CONFIG).elf

CFLAGS  := -O2 -march=$(ISA) -mabi=lp64 -mcmodel=medany -ffreestanding -fno-builtin \
           -fno-tree-loop-distribute-patterns -nostdlib -nostartfiles -Wall -Wextra \
           -Isrc/runtime -I$(GEN) $(EXTRA_CFLAGS)
LDFLAGS := -T src/runtime/linker.ld -Wl,--no-warn-rwx-segments

SRCS    := src/runtime/startup.S src/runtime/runtime.c src/runtime/model_data.S \
           src/bench/bench_layer.c src/kernels/gemm_$(CONFIG).c src/kernels/requant.c

.PHONY: all gen run test smoke ktest nofp clean
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
