// qmac8 - a custom RISC-V instruction, implemented as a Spike extension

//   qmac8 rd, rs1, rs2        "packed multiply-accumulate of 8 int8 pairs"

//   rs1 and rs2 are 64-bit registers. Each holds eight signed 8-bit values ("lanes"); lane 0 is the lowest
//   byte. The instruction multiplies lane i of rs1 with lane i of rs2 (never across lanes), adds the eight
//   products, and adds that sum to rd:

//       rd = rd + rs1.lane0 * rs2.lane0 + rs1.lane1 * rs2.lane1 + ... + rs1.lane7 * rs2.lane7

//   Example: rs1 = bytes 1,2,3,4,5,6,7,8   rs2 = bytes 1,1,1,1,1,1,1,1   rd = 100   ->   rd = 100 + 36 = 136

//   rd is both an input (the running sum) and the output, so a dot product over K elements is K/8 of these
//   instructions in a row. The add is a normal 64-bit wrap-around add, like ADD.

// WHY IT EXISTS
//   The baseline kernel needs about 7 instructions per multiply-accumulate (2 byte loads, 1 multiply, 1 add,
//   2 pointer increments, 1 branch). This instruction does 8 of them at once, fed by one 64-bit load per
//   operand. See results/b0.csv for the measurements that motivated it.

// ENCODING (32 bits, standard R-type format, in the "custom-0" opcode space reserved for custom instructions)

//   An instruction word w is a qmac8 if (w & MASK) == MATCH, with MATCH = 0x0000000b and MASK = 0xfe00707f
//   (the mask covers funct7, funct3 and the opcode; rd, rs1 and rs2 are free).
//   In C the compiler does not know the name, so tests use the generic assembler directive:
//       __asm__(".insn r 0x0b, 0, 0, %0, %1, %2" : "+r"(acc) : "r"(a), "r"(b));


#include "extension.h"   // extension_t, REGISTER_EXTENSION
#include "disasm.h"      // disasm_insn_t, arg_t, xpr_name (register names)
#include "trap.h"        // trap_illegal_instruction

#ifndef QMAC8_LANES
#define QMAC8_LANES 8    // always 8; the negative test builds a deliberately broken 7-lane library (-DQMAC8_LANES=7)
#endif

namespace {              // everything below is private to this library

const insn_bits_t QMAC8_MATCH = 0x0000000b;
const insn_bits_t QMAC8_MASK  = 0xfe00707f;

// Disassembly: how Spike prints the operands  e.g.  qmac8   a1, s1, s2
// Each small class turns one instruction field into its register name.

struct : public arg_t {
  std::string to_string(insn_t insn) const override { return xpr_name[insn.rd()]; }
} qrd;
struct : public arg_t {
  std::string to_string(insn_t insn) const override { return xpr_name[insn.rs1()]; }
} qrs1;
struct : public arg_t {
  std::string to_string(insn_t insn) const override { return xpr_name[insn.rs2()]; }
} qrs2;

// Execution: what the instruction does when the simulated CPU reaches it.
//   p     the simulated processor (gives access to its registers)
//   insn  the 32-bit instruction word (gives the register numbers rd, rs1, rs2)
//   pc    address of this instruction; the function returns the address of the NEXT instruction
//   logged  true when Spike runs with --log-commits, which also records every register write

reg_t qmac8_exec(processor_t *p, insn_t insn, reg_t pc, bool logged)
{
  state_t *s = p->get_state();
  reg_t a = s->XPR[insn.rs1()];          // XPR = the 32 integer registers x0..x31
  reg_t b = s->XPR[insn.rs2()];

  // Multiply lane i of a with lane i of b. (int8_t) takes the lowest 8 bits of the shifted value and treats
  // them as a signed number, which is exactly "lane i as int8"; the product of two int8 values fits easily.
  reg_t sum = 0;
  for (int i = 0; i < QMAC8_LANES; i++) {
    int64_t x = (int8_t)(a >> (8 * i));
    int64_t y = (int8_t)(b >> (8 * i));
    sum += (reg_t)(x * y);               // unsigned arithmetic: wraps around like the hardware ADD would
  }
  reg_t result = s->XPR[insn.rd()] + sum;   // rd is read as well as written

  if (logged)
    s->log_reg_write[insn.rd() << 4] = {result, 0};   // what --log-commits prints as "x11 0x..."
  s->XPR.write(insn.rd(), result);          // writing x0 is silently ignored by the register file
  return pc + 4;                            // a 32-bit instruction: the next one is 4 bytes further
}

// Spike keeps separate function pointers for normal and --log-commits runs, so there are two thin wrappers.
reg_t qmac8_fast(processor_t *p, insn_t insn, reg_t pc)   { return qmac8_exec(p, insn, pc, false); }
reg_t qmac8_logged(processor_t *p, insn_t insn, reg_t pc) { return qmac8_exec(p, insn, pc, true); }

#ifdef WITH_QREQ
// ---------------------------------------------------------------------------------------------------------------
// qreq rd, rs1, rs2    requantize one accumulator to int8 (config X2). Built only with -DWITH_QREQ (libqreq.so).
//
//   rs1[31:0]  int32 accumulator (the GEMM sum)
//   rs2[31:0]  int32 multiplier   (Q31 fixed point, per output channel)
//   rs2[36:32] right shift, 0..31 (per output channel)          rs2[63:37] ignored
//   rd         = clamp8( round_shift( high_mul(acc, mult), shift ) )   sign-extended to 64 bits
//
// It is exactly the arithmetic of src/kernels/requant.c (and scripts/quantize_w8a8.py):
//   high_mul    = round(acc * mult / 2^31), a tie rounds toward +infinity   (the gemmlowp "saturating rounding doubling high mul")
//   round_shift = x / 2^shift rounded to nearest, ties away from zero
//   clamp8      = limit to [-128, 127]
// Encoding: same R-type custom-0 opcode as qmac8, funct3 = 1.  MATCH = 0x0000100b, MASK = 0xfe00707f.
// ---------------------------------------------------------------------------------------------------------------
const insn_bits_t QREQ_MATCH = 0x0000100b;
const insn_bits_t QREQ_MASK  = 0xfe00707f;

reg_t qreq_exec(processor_t *p, insn_t insn, reg_t pc, bool logged)
{
  state_t *s = p->get_state();
  reg_t r1 = s->XPR[insn.rs1()];
  reg_t r2 = s->XPR[insn.rs2()];

  int64_t acc  = (int32_t)r1;                       // low 32 bits as a signed number
  int64_t mult = (int32_t)r2;
  int e = (int)((r2 >> 32) & 31);

  int64_t ab = acc * mult;                          // fits in 64 bits
  ab += (ab >= 0) ? (1LL << 30) : (1 - (1LL << 30));
  int64_t hm = (int32_t)(ab >= 0 ? (ab >> 31) : -((-ab) >> 31));

  int64_t mask = (1LL << e) - 1;                    // rounding right shift by e
  int64_t rem  = hm & mask;
  int64_t thr  = (mask >> 1) + (hm < 0);
  int64_t v    = (hm >> e) + (rem > thr);

  if (v > 127)  v = 127;                            // clamp to int8
  if (v < -128) v = -128;

  reg_t result = (reg_t)v;
  if (logged)
    s->log_reg_write[insn.rd() << 4] = {result, 0};
  s->XPR.write(insn.rd(), result);
  return pc + 4;
}
reg_t qreq_fast(processor_t *p, insn_t insn, reg_t pc)   { return qreq_exec(p, insn, pc, false); }
reg_t qreq_logged(processor_t *p, insn_t insn, reg_t pc) { return qreq_exec(p, insn, pc, true); }
#endif  // WITH_QREQ

// The instruction is defined for RV64 only. On RV32 or the reduced "E" register sets it is illegal.
reg_t qmac8_illegal(processor_t *, insn_t insn, reg_t) { throw trap_illegal_instruction(insn.bits()); }

// The extension object that Spike asks for the instruction table and the disassembly entries.
class qmac8_t : public extension_t
{
 public:
  #ifdef WITH_QREQ
  const char *name() const override { return "qreq"; }    // libqreq.so: qmac8 AND qreq, used as --extension=qreq
#else
  const char *name() const override { return "qmac8"; }   // the name used in --extension=qmac8
#endif

  std::vector<insn_desc_t> get_instructions(const processor_t &) override {
    std::vector<insn_desc_t> v = {
      // match, mask, then eight execute functions. Spike picks one by (xlen, E-register-set, logging):
      //   fast:    rv32i          rv64i       rv32e          rv64e
      //   logged:  rv32i          rv64i       rv32e          rv64e
      {QMAC8_MATCH, QMAC8_MASK,
       qmac8_illegal, qmac8_fast,   qmac8_illegal, qmac8_illegal,
       qmac8_illegal, qmac8_logged, qmac8_illegal, qmac8_illegal},
    };
#ifdef WITH_QREQ
    v.push_back({QREQ_MATCH, QREQ_MASK,
                 qmac8_illegal, qreq_fast,   qmac8_illegal, qmac8_illegal,
                 qmac8_illegal, qreq_logged, qmac8_illegal, qmac8_illegal});
#endif
    return v;
  }

  std::vector<disasm_insn_t *> get_disasms(const processor_t *) override {
    // mnemonic, match, mask, operand printers in order
    std::vector<disasm_insn_t *> v = {new disasm_insn_t("qmac8", QMAC8_MATCH, QMAC8_MASK, {&qrd, &qrs1, &qrs2})};
#ifdef WITH_QREQ
    v.push_back(new disasm_insn_t("qreq", QREQ_MATCH, QREQ_MASK, {&qrd, &qrs1, &qrs2}));
#endif
    return v;
  }
};

} 

// Makes the extension available to Spike under the name "qmac8" when the library is loaded.
#ifdef WITH_QREQ
REGISTER_EXTENSION(qreq, []() { static qmac8_t ext; return &ext; })
#else
REGISTER_EXTENSION(qmac8, []() { static qmac8_t ext; return &ext; })
#endif
