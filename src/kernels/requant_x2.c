#include "kernels.h"
#include "qreq.h"

/*
 X2: requantization with the qreq instruction. Same interface and same result as requant.c.

 qreq takes the multiplier and the shift in ONE 64-bit register, but the model data stores them as two separate
 arrays (the data format is the same for every configuration). So each call first packs them into a small table,
 once per output channel, and then every output value costs one load of the accumulator, one load of the packed
 parameters, one qreq and one store.

 N is processed in chunks of PACK_MAX channels
 */
#define PACK_MAX 1024
static uint64_t packed[PACK_MAX] __attribute__((aligned(8)));

void requant(const int32_t *acc, const int32_t *mult, const int32_t *shift,
             int8_t *out, int M, int N)
{
    for (int n0 = 0; n0 < N; n0 += PACK_MAX)
    {
        const int cn = (N - n0 < PACK_MAX) ? (N - n0) : PACK_MAX;

        for (int n = 0; n < cn; n++)
            packed[n] = ((uint64_t)(uint32_t)shift[n0 + n] << 32) | (uint32_t)mult[n0 + n];

        for (int m = 0; m < M; m++)
        {
            const int32_t *a = acc + m * N + n0;
            int8_t *o = out + m * N + n0;
            for (int n = 0; n < cn; n++)
                o[n] = (int8_t)qreq(a[n], packed[n]);
        }
    }
}
