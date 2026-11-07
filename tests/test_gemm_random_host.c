#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "kernels.h"

/*
 Host-side test (normal gcc on the PC) of whichever gemm_s8 is linked in (gemm_b0.c or gemm_b1.c).
 It compares gemm_s8 with a plain triple loop on random data, for many shapes, including odd M and odd N
 (the edge paths of the blocked B1 kernel) and extreme values (-128, 127). Exit code 0 = every shape matched.
 */

static uint64_t state = 0x2545F4914F6CDD1DULL;
static uint64_t rnd(void) { state ^= state << 13; state ^= state >> 7; state ^= state << 17; return state; }

int main(void)
{
    static int8_t x[8 * 40] __attribute__((aligned(8)));
    static int8_t w[8 * 40] __attribute__((aligned(8)));
    int32_t bias[8], got[64], want[64];
    int shapes = 0, bad = 0;

    for (int M = 1; M <= 7; M++)
        for (int N = 1; N <= 7; N++)
            for (int K = 8; K <= 40; K += 8)
                for (int trial = 0; trial < 20; trial++) {
                    for (int i = 0; i < M * K; i++) x[i] = (trial == 0) ? -128 : (int8_t)rnd();
                    for (int i = 0; i < N * K; i++) w[i] = (trial == 0) ? -128 : (trial == 1) ? 127 : (int8_t)rnd();
                    for (int i = 0; i < N; i++) bias[i] = (int32_t)(rnd() >> 40) - (1 << 22);

                    gemm_s8(x, w, bias, got, M, N, K);
                    for (int m = 0; m < M; m++)
                        for (int n = 0; n < N; n++) {
                            int32_t s = bias[n];
                            for (int k = 0; k < K; k++)
                                s += (int32_t)x[m * K + k] * (int32_t)w[n * K + k];
                            want[m * N + n] = s;
                        }
                    shapes++;
                    for (int i = 0; i < M * N; i++)
                        if (got[i] != want[i]) {
                            if (bad++ < 5) printf("MISMATCH M=%d N=%d K=%d index %d got %d want %d\n", M, N, K, i, got[i], want[i]);
                            break;
                        }
                }
    printf("random gemm test: %d cases, %d wrong  %s\n", shapes, bad, bad ? "FAIL" : "ok");
    return bad != 0;
}
