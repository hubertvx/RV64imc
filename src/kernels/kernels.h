#ifndef KERNELS_H
#define KERNELS_H

#include <stdint.h>

/*
 * Integer layer, split in two so their instruction costs can be measured separately:

    gemm_s8 : acc[m][n] = bias[n] + sum_k x[m][k] * w[n][k]      (int8 x int8 -> int32)
    requant : out[m][n] = clamp8( round_shift( high_mul(acc[m][n], mult[n]), shift[n] ) )

  x is M x K, w is N x K (K contiguous), both row-major and 8-byte aligned; K is a multiple of 8.
  Every configuration (B0 = plain C, X1 = with qmac8, ...) implements gemm_s8 with this signature.
 */
void gemm_s8(const int8_t *x, const int8_t *w, const int32_t *bias,
             int32_t *acc, int M, int N, int K);

void requant(const int32_t *acc, const int32_t *mult, const int32_t *shift,
             int8_t *out, int M, int N);

#endif
