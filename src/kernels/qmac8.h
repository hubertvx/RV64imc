#ifndef QMAC8_H
#define QMAC8_H

#include <stdint.h>

/*
 The qmac8 custom instruction 
      qmac8 rd, rs1, rs2:   rd = rd + sum_{i=0..7} int8(rs1 lane i) * int8(rs2 lane i)     (lane 0 = lowest byte)

 The compiler has no name for it, so the assembler directive .insn builds the 32-bit word:
  R-type (r), opcode 0x0b (custom-0), funct3 0, funct7 0, then rd, rs1, rs2.
    %0 is "+r": a register that is read and written (rd, the running sum)
    %1, %2 are "r": registers holding the two packed operands

  The instruction has no side effects besides writing rd, so the asm is not volatile and the compiler may schedule it like any other arithmetic instruction.
 */
static inline uint64_t qmac8(uint64_t acc, uint64_t a, uint64_t b)
{
    __asm__(".insn r 0x0b, 0, 0, %0, %1, %2" : "+r"(acc) : "r"(a), "r"(b));
    return acc;
}

/*
 Reads 8 consecutive int8 values with one 64-bit load. This is the packed operand that qmac8 expects.
 p must be 8-byte aligned. The may_alias attribute tells the compiler that reading int8 data through a
 64-bit type is intended
 */
static inline uint64_t load_int8x8(const int8_t *p)
{
    typedef uint64_t __attribute__((may_alias)) uint64_may_alias_t;
    return *(const uint64_may_alias_t *)p;
}

#endif
