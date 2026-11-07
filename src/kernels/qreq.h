#ifndef QREQ_H
#define QREQ_H

#include <stdint.h>

/*
 The requantization custom instruction (config X2), defined in src/isa/qmac8/qmac8.cc (built with -DWITH_QREQ):

    qreq rd, rs1, rs2:   rd = clamp8( round_shift( high_mul( int32(rs1), int32(rs2[31:0]) ), rs2[36:32] ) )

 rs1 is the int32 accumulator, rs2 packs the per-channel multiplier (low 32 bits) and right shift (bits 36:32).
 It does in one instruction what requant.c does in about 30. R-type, custom-0 opcode 0x0b, funct3 = 1.
 The result is a sign-extended int8 (-128..127), ready for a store byte.
 */
static inline int64_t qreq(int32_t acc, uint64_t param)
{
    int64_t rd;
    __asm__(".insn r 0x0b, 1, 0, %0, %1, %2" : "=r"(rd) : "r"((int64_t)acc), "r"(param));
    return rd;
}

#endif
