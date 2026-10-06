/*
 * Classify the dynamic instruction stream printed by `spike -l` (read on stdin).

 Only instructions whose PC lies in one of the ranges (the kernels' code, from `nm -S`) are
  counted, so startup code and printing are excluded. 

 A Spike trace line looks like:  core   0: 0x0000000080001426 (0x03e3033b) mulw    t1, t1, t5
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { LOAD, STORE, MUL, BRANCH, CUSTOM, ALU, NCLASS };
#define MAX_RANGES 16

static int in_list(const char *m, const char *const *list)
{
    for (; *list; list++)
        if (!strcmp(m, *list))
            return 1;
    return 0;
}

static const char *const loads[]   = {"lb", "lbu", "lh", "lhu", "lw", "lwu", "ld",
                                      "c.lw", "c.ld", "c.lwsp", "c.ldsp", 0};
static const char *const stores[]  = {"sb", "sh", "sw", "sd", "c.sw", "c.sd", "c.swsp", "c.sdsp", 0};
static const char *const muls[]    = {"mul", "mulh", "mulhsu", "mulhu", "mulw",
                                      "div", "divu", "divw", "divuw", "rem", "remu", "remw", "remuw", 0};
static const char *const branches[] = {"beq", "bne", "blt", "bge", "bltu", "bgeu",
                                      "beqz", "bnez", "blez", "bgez", "bltz", "bgtz", "bgt", "ble",
                                      "bgtu", "bleu", "c.beqz", "c.bnez", "c.j", "c.jal", "c.jr",
                                      "c.jalr", "jal", "jalr", "j", "jr", "ret", 0};
static const char *const customs[] = {"qmac8", "qreq", 0};

int main(int argc, char **argv)
{
    char names[MAX_RANGES][64];
    unsigned long long lo[MAX_RANGES], hi[MAX_RANGES], cnt[MAX_RANGES][NCLASS] = {{0}}, tot[MAX_RANGES] = {0};
    int n = argc - 1;

    if (n < 1 || n > MAX_RANGES) {
        fprintf(stderr, "usage: instr_mix name=LO-HI ...\n");
        return 2;
    }
    for (int i = 0; i < n; i++) {
        char *eq = strchr(argv[i + 1], '='), *dash = eq ? strchr(eq, '-') : 0;
        if (!eq || !dash) {
            fprintf(stderr, "bad range: %s\n", argv[i + 1]);
            return 2;
        }
        snprintf(names[i], sizeof names[i], "%.*s", (int)(eq - argv[i + 1]), argv[i + 1]);
        lo[i] = strtoull(eq + 1, 0, 16);
        hi[i] = strtoull(dash + 1, 0, 16);
    }

    char line[512];
    while (fgets(line, sizeof line, stdin)) {
        if (strncmp(line, "core", 4))
            continue;
        char *p = strstr(line, ": 0x");
        if (!p)
            continue;
        unsigned long long pc = strtoull(p + 2, &p, 16);
        int r = -1;
        for (int i = 0; i < n; i++)
            if (pc >= lo[i] && pc < hi[i]) { r = i; break; }
        if (r < 0)
            continue;
        p = strchr(p, ')');
        if (!p)
            continue;
        p++;
        while (*p == ' ')
            p++;
        char mn[32];
        int k = 0;
        while (*p && *p != ' ' && *p != '\n' && k < 31)
            mn[k++] = *p++;
        mn[k] = 0;

        int c = ALU;
        if (in_list(mn, loads)) c = LOAD;
        else if (in_list(mn, stores)) c = STORE;
        else if (in_list(mn, muls)) c = MUL;
        else if (in_list(mn, branches)) c = BRANCH;
        else if (in_list(mn, customs)) c = CUSTOM;
        cnt[r][c]++;
        tot[r]++;
    }
    for (int i = 0; i < n; i++)
        printf("%s %llu %llu %llu %llu %llu %llu %llu\n", names[i], tot[i], cnt[i][LOAD], cnt[i][STORE],
               cnt[i][MUL], cnt[i][BRANCH], cnt[i][CUSTOM], cnt[i][ALU]);
    return 0;
}
