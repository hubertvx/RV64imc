#include "runtime.h"

/*
Smoke test for the bare-metal runtime (startup.S, linker.ld, runtime.c, runtime.h)
It needs nothing else: no kernels, no model data. Exit code 0 = everything works.
 */

static uint64_t zeroed[64];                              /* .bss: startup.S must clear it */
static uint64_t initialized = 0x1122334455667788ULL;     /* .data: must be loaded from the ELF */

int main(void)
{
    int fail = 0;

    rt_puts("hello from RV64IMC\n");                     /* console output works */
    rt_report("ZERO", 0);                                /* number printing: edge cases */
    rt_report("ANSWER", 42);
    rt_report("MAX_U64", 18446744073709551615ULL);

    for (int i = 0; i < 64; i++)
        fail |= (zeroed[i] != 0);
    fail |= (initialized != 0x1122334455667788ULL);

    uint64_t t0 = rt_instret();                          /* instruction counter works */
    for (volatile int i = 0; i < 1000; i++)
        ;
    uint64_t dt = rt_instret() - t0;
    rt_report("INSTRET_1000_LOOP", dt);
    fail |= (dt < 1000);                                 /* at least one instruction per iteration */

    rt_puts(fail ? "FAIL\n" : "PASS\n");
    return fail;                                         /* becomes the exit code */
}
