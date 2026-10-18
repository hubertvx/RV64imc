#include "runtime.h"

/*
 Verification of the qmac8 instruction (see src/isa/qmac8/qmac8.cc for the definition).

 qmac8 rd, rs1, rs2:   rd = rd + sum_{i=0..7} int8(rs1 lane i) * int8(rs2 lane i)     (lane 0 = lowest byte)

 WHAT IS CHECKED (the program counts its checks and prints PASS / FAIL; exit code 0 = every check passed)
  - directed cases: expected values calculated BY HAND, not by the C reference below, so a bug in the
    reference cannot hide a bug in the instruction (zeros, lanes that must not mix, extremes -128/127,
   negative numbers, 64-bit wrap-around)
  - register overlaps: the same register used as rd and rs1, rd and rs2, rs1 and rs2, all three; and rd = x0 (the result must be discarded and x0 must stay 0)
  - accumulation over several calls in a row, the way the kernel will use it
  - 10,000 random operand triples compared with the plain C reference
  - exhaustive: all 256 x 256 int8 pairs in each of the 8 lanes (524,288 checks), expected value from plain C
 */

/*
 The instruction under test. The compiler has no mnemonic for it, so the assembler directive .insn builds the
 32-bit word: R-type (r), opcode 0x0b (custom-0), funct3 0, funct7 0, then rd, rs1, rs2.
 operand %0 is "+r": a register that is both read and written (rd is the accumulator).
 operands %1 and %2 are "r": read-only registers holding the packed operands.
 */
static inline uint64_t qmac8(uint64_t acc, uint64_t a, uint64_t b)
{
    __asm__ volatile(".insn r 0x0b, 0, 0, %0, %1, %2" : "+r"(acc) : "r"(a), "r"(b));
    return acc;
}

/* plain C reference of the same semantics (wrap-around 64-bit add) */
static uint64_t ref(uint64_t acc, uint64_t a, uint64_t b)
{
    uint64_t sum = 0;
    for (int i = 0; i < 8; i++)
        sum += (uint64_t)((int64_t)(int8_t)(a >> (8 * i)) * (int64_t)(int8_t)(b >> (8 * i)));
    return acc + sum;
}

static void put_hex(uint64_t v)
{
    static const char d[] = "0123456789abcdef";
    rt_puts("0x");
    for (int s = 60; s >= 0; s -= 4)
        rt_putc(d[(v >> s) & 0xf]);
}

static uint64_t checks, failed;

static void check(const char *what, uint64_t got, uint64_t expected)
{
    checks++;
    if (got == expected)
        return;
    failed++;
    if (failed <= 10) {
        rt_puts("FAILED: "); rt_puts(what);
        rt_puts("  got "); put_hex(got);
        rt_puts("  expected "); put_hex(expected);
        rt_putc('\n');
    }
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t rng(void)                  /* xorshift64: deterministic, so a failure can be reproduced */
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

int main(void)
{
    /*  directed, expected values calculated by hand */
    check("zeros",                      qmac8(0, 0, 0), 0);
    check("acc passes through",         qmac8(123, 0, 0x0102030405060708ULL), 123);
    check("8 lanes of 1*1",             qmac8(0, 0x0101010101010101ULL, 0x0101010101010101ULL), 8);
    check("lane 0 only: 3*4",           qmac8(0, 3, 4), 12);
    check("lanes do not mix: 3 vs lane1", qmac8(0, 3, 0x400), 0);
    check("lane 7 only: 5*6",           qmac8(0, 0x0500000000000000ULL, 0x0600000000000000ULL), 30);
    check("-1 * 1, 8 lanes",            qmac8(0, ~0ULL, 0x0101010101010101ULL), (uint64_t)-8LL);
    check("-1 * -1, 8 lanes",           qmac8(0, ~0ULL, ~0ULL), 8);
    check("-128 * -128, 8 lanes",       qmac8(0, 0x8080808080808080ULL, 0x8080808080808080ULL), 131072);
    check("-128 * 127, 8 lanes",        qmac8(0, 0x8080808080808080ULL, 0x7F7F7F7F7F7F7F7FULL), (uint64_t)-130048LL);
    check("127 * 127, 8 lanes",         qmac8(0, 0x7F7F7F7F7F7F7F7FULL, 0x7F7F7F7F7F7F7F7FULL), 129032);
    /* a lanes 1,-2,3,-4,5,-6,7,-8 times 1: sum = -4, plus acc 100 */
    check("mixed signs + acc 100",      qmac8(100, 0xF807FA05FC03FE01ULL, 0x0101010101010101ULL), 96);
    check("negative acc -1000 + 36",    qmac8((uint64_t)-1000LL, 0x0807060504030201ULL, 0x0101010101010101ULL), (uint64_t)-964LL);
    check("1..8 squared = 204",         qmac8(0, 0x0807060504030201ULL, 0x0807060504030201ULL), 204);
    check("64-bit wrap-around",         qmac8(0x7FFFFFFFFFFFFFFFULL, 1, 1), 0x8000000000000000ULL);

    /* register overlaps (same physical register used for several operands) */
    {
        uint64_t x = 0x0807060504030201ULL, y = 0x0101010101010101ULL;
        uint64_t r;

        r = x; __asm__ volatile(".insn r 0x0b, 0, 0, %0, %0, %1" : "+r"(r) : "r"(y));
        check("rd == rs1: x + dot(x,y)", r, ref(x, x, y));              /* 0x0807.. + 36 */

        r = x; __asm__ volatile(".insn r 0x0b, 0, 0, %0, %1, %0" : "+r"(r) : "r"(y));
        check("rd == rs2: x + dot(y,x)", r, ref(x, y, x));

        r = 1000; __asm__ volatile(".insn r 0x0b, 0, 0, %0, %1, %1" : "+r"(r) : "r"(x));
        check("rs1 == rs2: 1000 + 204", r, 1204);                      /* sum of squares 1..8 */

        r = x; __asm__ volatile(".insn r 0x0b, 0, 0, %0, %0, %0" : "+r"(r));
        check("rd == rs1 == rs2: x + dot(x,x)", r, ref(x, x, x));

        uint64_t z;
        __asm__ volatile(".insn r 0x0b, 0, 0, x0, %0, %1" : : "r"(x), "r"(y));   /* rd = x0: result discarded */
        __asm__ volatile("mv %0, x0" : "=r"(z));
        check("rd == x0 leaves x0 == 0", z, 0);
    }

    /*  accumulation over several calls  */
    {
        uint64_t acc = 0, expected = 0;
        for (int i = 1; i <= 16; i++) {
            uint64_t a = 0x0101010101010101ULL * (uint64_t)i;      /* every lane = i */
            acc = qmac8(acc, a, 0x0101010101010101ULL);            /* adds 8*i */
            expected += 8 * (uint64_t)i;
        }
        check("accumulate 16 calls: 8*(1+..+16)", acc, expected);   /* 1088 */
    }

    /*   random operands against the C reference */
    uint64_t random_failed_before = failed;
    for (int i = 0; i < 10000; i++) {
        uint64_t acc = rng(), a = rng(), b = rng();
        check("random triple", qmac8(acc, a, b), ref(acc, a, b));
    }
    uint64_t random_failed = failed - random_failed_before;

    /*
     Exhaustive: every int8 x int8 pair (256 * 256 = 65,536) in every lane (8), so 524,288 checks.
     */
    uint64_t exhaustive_failed_before = failed;
    for (int lane = 0; lane < 8; lane++) {
        for (int x = -128; x <= 127; x++) {
            for (int y = -128; y <= 127; y++) {
                uint64_t a = (uint64_t)(uint8_t)x << (8 * lane);
                uint64_t b = (uint64_t)(uint8_t)y << (8 * lane);
                check("exhaustive lane product", qmac8(0, a, b), (uint64_t)(int64_t)(x * y));
            }
        }
    }

    rt_report("CHECKS", checks);
    rt_report("FAILED", failed);
    rt_report("RANDOM_FAILED", random_failed);
    rt_report("EXHAUSTIVE_FAILED", failed - exhaustive_failed_before);
    rt_puts(failed ? "FAIL\n" : "PASS\n");
    return failed != 0;
}
