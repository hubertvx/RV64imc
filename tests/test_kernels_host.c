#include <stdio.h>
#include "kernels.h"

/*
 Host-side test of the kernels (runs on the PC with normal gcc, no RISC-V) 

 Every expected value is small enough to calculate by hand. Exit code 0 = all pas.
 */

static int check(const char *what, long got, long expect)
{
    int ok = (got == expect);
    printf("%-34s got %6ld expect %6ld  %s\n", what, got, expect, ok ? "ok" : "WRONG");
    return !ok;
}

int main(void)
{
    int bad = 0;

    /* gemm_s8: 1 row, 2 outputs, K = 8.   x = 1..8, w0 = all +1, w1 = all -1, bias = {10, 0}
     *   out0 = 10 + (1+2+...+8) = 46        out1 = 0 - 36 = -36                              */
    int8_t  x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int8_t  w[16] = {1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1};
    int32_t bias[2] = {10, 0}, acc[2];
    gemm_s8(x, w, bias, acc, 1, 2, 8);
    bad |= check("gemm out0 = 10 + 36", acc[0], 46);
    bad |= check("gemm out1 = 0 - 36", acc[1], -36);

    /* gemm_s8 with 2 rows (M = 2): the second row is x*2, so its sums are 2*36 plus bias */
    int8_t  x2[16] = {1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 8, 10, 12, 14, 16};
    int32_t acc2[4];
    gemm_s8(x2, w, bias, acc2, 2, 2, 8);
    bad |= check("gemm row1 out0 = 10 + 72", acc2[2], 82);
    bad |= check("gemm row1 out1 = 0 - 72", acc2[3], -72);

    /* extreme values: -128 * -128 * 8 = 131072 (largest product) */
    int8_t  xm[8] = {-128, -128, -128, -128, -128, -128, -128, -128};
    int8_t  wm[8] = {-128, -128, -128, -128, -128, -128, -128, -128};
    int32_t zero[1] = {0}, accm[1];
    gemm_s8(xm, wm, zero, accm, 1, 1, 8);
    bad |= check("gemm (-128*-128)*8", accm[0], 131072);

    /* requant with mult = 2^30 (= 0.5 in Q31) and shift = 1: value * 0.5 / 2 = value / 4,
     * rounded to nearest, ties away from zero.  46/4 = 11.5 -> 12,  -36/4 = -9              */
    int32_t mult[1] = {1 << 30}, shift[1] = {1};
    int32_t in1[1];
    int8_t  out1[1];
    in1[0] = 46;      requant(in1, mult, shift, out1, 1, 1);  bad |= check("requant 46/4 = 11.5 -> 12", out1[0], 12);
    in1[0] = -46;     requant(in1, mult, shift, out1, 1, 1);  bad |= check("requant -46/4 = -11.5 -> -12", out1[0], -12);
    in1[0] = -36;     requant(in1, mult, shift, out1, 1, 1);  bad |= check("requant -36/4 = -9", out1[0], -9);
    /* Rounding happens in two stages (gemmlowp/TFLite): 45*0.5 = 22.5 -> 23, then 23/2 = 11.5 -> 12.
     * So the result is not always the correctly rounded 45/4 = 11.25 -> 11. It must equal numpy's. */
    in1[0] = 45;      requant(in1, mult, shift, out1, 1, 1);  bad |= check("requant 45: two-stage rounding", out1[0], 12);
    in1[0] = -45;     requant(in1, mult, shift, out1, 1, 1);  bad |= check("requant -45: tie rounds up first", out1[0], -11);
    in1[0] = 100000;  requant(in1, mult, shift, out1, 1, 1);  bad |= check("requant saturates high", out1[0], 127);
    in1[0] = -100000; requant(in1, mult, shift, out1, 1, 1);  bad |= check("requant saturates low", out1[0], -128);

    /* a different multiplier per output channel (n index): channel 0 keeps value/4, channel 1 is value/8 */
    int32_t multc[2] = {1 << 30, 1 << 30}, shiftc[2] = {1, 2};
    int32_t inc[2] = {64, 64};
    int8_t  outc[2];
    requant(inc, multc, shiftc, outc, 1, 2);
    bad |= check("per-channel: 64/4", outc[0], 16);
    bad |= check("per-channel: 64/8", outc[1], 8);

    printf(bad ? "FAIL\n" : "PASS\n");
    return bad;
}
