/* Embed the exported model data (scripts/export_c.py) into the ELF, 8-byte aligned. */
    .section .rodata
    .balign 8
    .globl weights_blob
weights_blob:
    .incbin "weights.bin"

    .balign 8
    .globl vectors_blob
vectors_blob:
    .incbin "vectors.bin"
