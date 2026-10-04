#include "kernels.h"

// Same arithmetic as scripts/quantize_w8a8.py , integers only.

// saturating rounding doubling high multiply: round(a * b / 2^31) to nearest.
// An exact tie rounds up (22.5 -> 23, -22.5 -> -22): this is what gemmlowp/TFLite do. 
static inline int32_t high_mul(int32_t a, int32_t b)
{
    int64_t ab = (int64_t)a * b;
    ab += (ab >= 0) ? (1LL << 30) : (1 - (1LL << 30));
    return (int32_t)(ab >= 0 ? (ab >> 31) : -((-ab) >> 31)); 
}

//  arithmetic right shift by e with rounding to nearest, ties away from zero
static inline int32_t round_shift(int32_t x, int e)
{
    int32_t mask = (1 << e) - 1;
    int32_t rem = x & mask;
    int32_t thr = (mask >> 1) + (x < 0);
    return (x >> e) + (rem > thr);
}

void requant(const int32_t *acc, const int32_t *mult, const int32_t *shift,
             int8_t *out, int M, int N)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t v = round_shift(high_mul(acc[m * N + n], mult[n]), shift[n]);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            out[m * N + n] = (int8_t)v;
        }
    }
}
