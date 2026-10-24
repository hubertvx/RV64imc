#include "kernels.h"
#include "qmac8.h"

/*
 X1: B0 plus the qmac8 instruction. The loop structure is the same as in gemm_b0.c (no unrolling, no
 blocking), so the difference between B0 and X1 measures the instruction

 The B0 inner loop does one multiply-accumulate per iteration from two byte loads. Here each iteration
 loads 8 bytes of x and 8 bytes of w with two 64-bit loads and does 8 multiply-accumulates with one qmac8, so the inner loop runs K / 8 times.

 */
void gemm_s8(const int8_t *x, const int8_t *w, const int32_t *bias,
             int32_t *acc, int M, int N, int K)
{
    for (int m = 0; m < M; m++) {
        const int8_t *xr = x + m * K;
        for (int n = 0; n < N; n++) {
            const int8_t *wr = w + n * K;

            // qmac8 adds into a 64-bit register: start from the bias, sign-extended 
            uint64_t sum = (uint64_t)(int64_t)bias[n];
            for (int k = 0; k < K; k += 8)
                sum = qmac8(sum, load_int8x8(xr + k), load_int8x8(wr + k));

        //    the true sum always fits in 32 bits
            acc[m * N + n] = (int32_t)sum;
        }
    }
}
