OUTPUT_ARCH(riscv)
ENTRY(_start)

SECTIONS
{
  . = 0x80000000;
  .text.init : { *(.text.init) }

  . = ALIGN(0x1000);
  .tohost : { *(.tohost) }

  .text : { *(.text .text.*) }
  .rodata : { *(.rodata .rodata.*) *(.srodata .srodata.*) }
  .data : { *(.data .data.*) *(.sdata .sdata.*) }

  . = ALIGN(8);
  _bss_start = .;
  .bss : { *(.sbss .sbss.*) *(.bss .bss.*) *(COMMON) }
  . = ALIGN(8);
  _bss_end = .;

  . = ALIGN(16);
  . += 0x10000;
  _stack_top = .;
}
