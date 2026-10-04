#include "kernels.h"

/* B0: no unrolling, no blocking, one byte load per operand per MAC. */
void gemm_s8(const int8_t *x, const int8_t *w, const int32_t *bias,
             int32_t *acc, int M, int N, int K)
{
    for (int m = 0; m < M; m++) {
        const int8_t *xr = x + m * K;
        for (int n = 0; n < N; n++) {
            const int8_t *wr = w + n * K;
            int32_t sum = bias[n];
            for (int k = 0; k < K; k++)
                sum += (int32_t)xr[k] * (int32_t)wr[k];
            acc[m * N + n] = sum;
        }
    }
}
