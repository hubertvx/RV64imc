#include "kernels.h"

/*
 B1: B0 with software optimization, loop unrolling.

 The only change from gemm_b0.c: the K loop handles 8 elements per pass instead of 1. The loop counter update and the
 branch are paid once per 8 MACs instead of once per MAC, and the loads use fixed offsets (xr[k + 0] ... xr[k + 7])
 instead of moving two pointers on every MAC. K is always a multiple of 8 (see kernels.h), so no leftover loop is needed.

 Each MAC still costs 2 byte loads, 1 multiply and 1 add: RV64IMC has no SIMD, so unrolling cannot reduce those.
 */

void gemm_s8(const int8_t *x, const int8_t *w, const int32_t *bias,
             int32_t *acc, int M, int N, int K)
{
    for (int m = 0; m < M; m++)
    {
        const int8_t *xr = x + m * K;
        for (int n = 0; n < N; n++)
        {
            const int8_t *wr = w + n * K;
            int32_t sum = bias[n];
            for (int k = 0; k < K; k += 8)
            {
                sum += (int32_t)xr[k + 0] * (int32_t)wr[k + 0];
                sum += (int32_t)xr[k + 1] * (int32_t)wr[k + 1];
                sum += (int32_t)xr[k + 2] * (int32_t)wr[k + 2];
                sum += (int32_t)xr[k + 3] * (int32_t)wr[k + 3];
                sum += (int32_t)xr[k + 4] * (int32_t)wr[k + 4];
                sum += (int32_t)xr[k + 5] * (int32_t)wr[k + 5];
                sum += (int32_t)xr[k + 6] * (int32_t)wr[k + 6];
                sum += (int32_t)xr[k + 7] * (int32_t)wr[k + 7];
            }
            acc[m * N + n] = sum;
        }
    }
}
