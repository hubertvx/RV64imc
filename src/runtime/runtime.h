#ifndef RT_H
#define RT_H

#include <stdint.h>

/* bare-metal runtime for Spike: M-mode, no libc, console and exit through HTIF. */

void rt_putc(char c);
void rt_puts(const char *s);
void rt_report(const char *key, uint64_t value);   
void runtime_exit(int code) __attribute__((noreturn));  

/* retired-instruction counter (M-mode CSR). */
static inline uint64_t rt_instret(void)
{
    uint64_t x;
    __asm__ volatile("csrr %0, minstret" : "=r"(x));
    return x;
}

#endif