#include "runtime.h"
#include "qreq.h"

/*
 Verification of the qreq instruction (see src/isa/qmac8/qmac8.cc, built with -DWITH_QREQ).

 qreq rd, rs1, rs2:   rd = clamp8( round_shift( high_mul( int32(rs1), int32(rs2[31:0]) ), rs2[36:32] ) )

 The program counts its checks and prints PASS / FAIL; exit code 0 = every check passed
  - directed cases whose expected values are worked out by hand: rounding of ties up and down, negative values
    (high multiply: tie toward +infinity; shift: tie away from zero), clamping at -128 and 127, shift 0, shift 31, upper bits of rs1/rs2 ignored
  - register overlaps: rd == rs1, rd == rs2, rd = x0 (the result must be discarded)
  - 1,000,000 random (acc, mult, shift) triples against the C reference of src/kernels/requant.c written out below
  - a sweep of every accumulator in -70000..70000 for a fixed multiplier and each shift 0..31
 */

static inline int64_t qreq_ovl(int which, int64_t acc, uint64_t param) /* register overlap variants */
{
    int64_t r;
    if (which == 0)
    {
        r = acc;
        __asm__ volatile(".insn r 0x0b, 1, 0, %0, %0, %1" : "+r"(r) : "r"(param));
    }
    else
    {
        r = (int64_t)param;
        __asm__ volatile(".insn r 0x0b, 1, 0, %0, %1, %0" : "+r"(r) : "r"(acc));
    }
    return r;
}

/* reference: the arithmetic of requant.c, with the same operations */
static int32_t high_mul(int32_t a, int32_t b)
{
    int64_t ab = (int64_t)a * b;
    ab += (ab >= 0) ? (1LL << 30) : (1 - (1LL << 30));
    return (int32_t)(ab >= 0 ? (ab >> 31) : -((-ab) >> 31));
}
static int32_t round_shift(int32_t x, int e)
{
    int32_t mask = (int32_t)((1LL << e) - 1);
    int32_t rem = x & mask;
    int32_t thr = (mask >> 1) + (x < 0);
    return (x >> e) + (rem > thr);
}
static int64_t ref(int32_t acc, int32_t mult, int e)
{
    int32_t v = round_shift(high_mul(acc, mult), e);
    if (v > 127)
        v = 127;
    if (v < -128)
        v = -128;
    return v;
}
static uint64_t pack(int32_t mult, int e) { return ((uint64_t)(uint32_t)e << 32) | (uint32_t)mult; }

static uint64_t checks, failed;
static void put_hex(uint64_t v)
{
    static const char d[] = "0123456789abcdef";
    rt_puts("0x");
    for (int s = 60; s >= 0; s -= 4)
        rt_putc(d[(v >> s) & 0xf]);
}
static void check(const char *what, int64_t got, int64_t expected)
{
    checks++;
    if (got == expected)
        return;
    failed++;
    if (failed <= 10)
    {
        rt_puts("FAILED: ");
        rt_puts(what);
        rt_puts("  got ");
        put_hex((uint64_t)got);
        rt_puts("  expected ");
        put_hex((uint64_t)expected);
        rt_putc('\n');
    }
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t rng(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

int main(void)
{
    const int32_t HALF = 1 << 30; /* multiplier 0.5 in Q31: acc * mult / 2^31 = acc / 2 */

    /* directed, expected values worked out by hand: with mult = 2^30 the high multiply gives acc / 2 */
    check("zero", qreq(0, pack(HALF, 0)), 0);
    check("acc 100, x0.5, shift 0 -> 50", qreq(100, pack(HALF, 0)), 50);
    check("acc 100, x0.5, shift 1 -> 25", qreq(100, pack(HALF, 1)), 25);
    check("acc 3, x0.5 = 1.5 -> 2 (tie up)", qreq(3, pack(HALF, 0)), 2);
    check("acc -3, x0.5 = -1.5 -> -1 (tie up)", qreq(-3, pack(HALF, 0)), -1);
    check("acc 5, x0.5 = 2.5 -> 3", qreq(5, pack(HALF, 0)), 3);
    check("acc -5, x0.5 = -2.5 -> -2", qreq(-5, pack(HALF, 0)), -2);
    check("acc 1, x0.5 = 0.5 -> 1", qreq(1, pack(HALF, 0)), 1);
    check("acc -1, x0.5 = -0.5 -> 0", qreq(-1, pack(HALF, 0)), 0);
    check("acc -2, x0.5 = -1", qreq(-2, pack(HALF, 0)), -1);
    /* the high multiply rounds a tie toward +infinity; the rounding SHIFT rounds a tie away from zero: */
    check("shift tie: 3 >> 1 = 1.5 -> 2", qreq(6, pack(HALF, 1)), 2);      /* hm = 3, 3/2 = 1.5 */
    check("shift tie: -3 >> 1 = -1.5 -> -2", qreq(-6, pack(HALF, 1)), -2); /* hm = -3, -3/2 = -1.5 */
    check("clamp high: 1000 -> 127", qreq(2000, pack(HALF, 0)), 127);
    check("clamp low: -1000 -> -128", qreq(-2000, pack(HALF, 0)), -128);
    check("exactly 127", qreq(254, pack(HALF, 0)), 127);
    check("exactly -128", qreq(-256, pack(HALF, 0)), -128);
    check("just over: 128 -> 127", qreq(256, pack(HALF, 0)), 127);
    check("shift 7: 1024/2/128 = 4", qreq(1024, pack(HALF, 7)), 4);
    check("shift 31 of small value is 0", qreq(1000, pack(HALF, 31)), 0);
    check("shift 31 of negative is 0", qreq(-1000, pack(HALF, 31)), 0);
    check("mult 0 -> 0", qreq(123456, pack(0, 3)), 0);
    check("negative multiplier", qreq(100, pack(-HALF, 0)), -50);

    /* fields that must be ignored: rs1 above bit 31, rs2 above bit 36 */
    {
        int64_t a_hi = (int64_t)(((uint64_t)0xDEADBEEFULL << 32) | 100u); /* low 32 bits = 100 */
        check("rs1 upper bits ignored", qreq_ovl(0, a_hi, pack(HALF, 0)), 50);
        uint64_t p_hi = pack(HALF, 0) | (0x7FFFFFULL << 37); /* junk in rs2[63:37] */
        int64_t r;
        __asm__ volatile(".insn r 0x0b, 1, 0, %0, %1, %2" : "=r"(r) : "r"((int64_t)100), "r"(p_hi));
        check("rs2 bits 63:37 ignored", r, 50);
    }

    /* register overlaps, and rd = x0 */
    {
        check("rd == rs1", qreq_ovl(0, 100, pack(HALF, 0)), 50);
        check("rd == rs2", qreq_ovl(1, 100, pack(HALF, 0)), 50);
        int64_t z;
        __asm__ volatile(".insn r 0x0b, 1, 0, x0, %0, %1" : : "r"((int64_t)100), "r"(pack(HALF, 0)));
        __asm__ volatile("mv %0, x0" : "=r"(z));
        check("rd == x0 leaves x0 == 0", z, 0);
    }

    /* random triples: accumulator in the range a real GEMM produces (23 signed bits) and beyond, any valid
       multiplier, any shift 0..30 (the model uses 7..15) */
    uint64_t before = failed;
    for (int i = 0; i < 1000000; i++)
    {
        int32_t acc = (int32_t)(rng() >> 33) - (1 << 30);
        if (i & 1)
            acc >>= 8;                                        /* also many small accumulators */
        int32_t mult = (int32_t)((rng() >> 33) | (1u << 30)); /* in [2^30, 2^31) like the model */
        if (i % 7 == 0)
            mult = (int32_t)(rng() >> 32); /* and arbitrary int32 */
        int e = (int)(rng() % 31);
        check("random", qreq(acc, pack(mult, e)), ref(acc, mult, e));
    }
    uint64_t random_failed = failed - before;

    /* sweep: every accumulator near zero, every shift */
    before = failed;
    for (int e = 0; e <= 31; e++)
        for (int32_t a = -70000; a <= 70000; a++)
            check("sweep", qreq(a, pack(1100000000, e)), ref(a, 1100000000, e));
    uint64_t sweep_failed = failed - before;

    rt_report("CHECKS", checks);
    rt_report("FAILED", failed);
    rt_report("RANDOM_FAILED", random_failed);
    rt_report("SWEEP_FAILED", sweep_failed);
    rt_puts(failed ? "FAIL\n" : "PASS\n");
    return failed != 0;
}
