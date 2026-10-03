#include "runtime.h"

volatile uint64_t tohost   __attribute__((section(".tohost"), aligned(64)));
volatile uint64_t fromhost __attribute__((section(".tohost"), aligned(64)));

void rt_putc(char c)
{
    tohost = (1ULL << 56) | (1ULL << 48) | (uint8_t)c;
    while (tohost);
    fromhost = 0;
}

void rt_puts(const char *s)
{
    while (*s)
        rt_putc(*s++);
}

void rt_report(const char *key, uint64_t v)
{
    char buf[21];
    int i = 20;

    buf[i] = 0;
    do {
        buf[--i] = '0' + (v % 10);
        v /= 10;
    } while (v);

    rt_puts(key);
    rt_puts(": ");
    rt_puts(&buf[i]);
    rt_putc('\n');
}

void runtime_exit(int code)
{
    tohost = ((uint64_t)code << 1) | 1;
    while(1);
}
