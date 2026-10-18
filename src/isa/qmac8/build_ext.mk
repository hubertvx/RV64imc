# Build the qmac8 Spike extension (a shared library).
# The library is compiled against the Spike source copy in src/isa/riscv-isa-sim and its build directory,

HERE   := $(dir $(lastword $(MAKEFILE_LIST)))
SPIKE  ?= $(HERE)../riscv-isa-sim
BUILD  ?= $(SPIKE)/build
OUT    ?= build/libqmac8.so
CXX    ?= g++
EXTRA  ?=

INCLUDES := -iquote $(BUILD) -I$(SPIKE) -iquote $(SPIKE)/riscv -iquote $(SPIKE)/fesvr -iquote $(SPIKE)/softfloat \
            -iquote $(SPIKE)/disasm -iquote $(SPIKE)/fdt -iquote $(SPIKE)/customext

$(OUT): $(HERE)qmac8.cc
	@mkdir -p $(dir $@)
	$(CXX) -std=c++2a -O2 -fPIC -shared -Wall -Wextra -Wno-unused-parameter $(INCLUDES) $(EXTRA) -o $@ $<
