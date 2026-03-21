#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/common/wave_execution_state.h"
#include "lib/sim/isa/gfx1201/architecture_profile.h"
#include "lib/sim/isa/gfx1201/binary_decoder.h"
#include "lib/sim/isa/gfx1201/interpreter.h"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }
  return true;
}

constexpr std::uint32_t SetBits(std::uint32_t word,
                                std::uint32_t value,
                                std::uint32_t bit_offset,
                                std::uint32_t bit_count) {
  const std::uint32_t mask =
      (bit_count == 32) ? 0xffffffffu : ((1u << bit_count) - 1u);
  return word | ((value & mask) << bit_offset);
}

constexpr std::uint32_t MakeSopp(std::uint32_t op, std::uint32_t simm16 = 0) {
  std::uint32_t word = 0;
  word = SetBits(word, 0x17f, 23, 9);
  word = SetBits(word, op, 16, 7);
  word = SetBits(word, simm16, 0, 16);
  return word;
}

constexpr std::uint32_t MakeSopk(std::uint32_t op,
                                 std::uint32_t sdst,
                                 std::uint32_t simm16) {
  std::uint32_t word = 0;
  word = SetBits(word, 0xb, 28, 4);
  word = SetBits(word, op, 23, 5);
  word = SetBits(word, sdst, 16, 7);
  word = SetBits(word, simm16, 0, 16);
  return word;
}

std::array<std::uint32_t, 2> MakeSmem(std::uint32_t op,
                                      std::uint32_t sdata,
                                      std::uint32_t sbase_start,
                                      bool imm,
                                      std::uint32_t offset_or_soffset,
                                      bool soffset_en = false) {
  std::uint64_t word = 0;
  word |= static_cast<std::uint64_t>(0x30u) << 26;
  word |= static_cast<std::uint64_t>(sbase_start >> 1);
  word |= static_cast<std::uint64_t>(sdata) << 6;
  word |= static_cast<std::uint64_t>(soffset_en ? 1u : 0u) << 14;
  word |= static_cast<std::uint64_t>(imm ? 1u : 0u) << 17;
  word |= static_cast<std::uint64_t>(op) << 18;
  if (imm) {
    word |= static_cast<std::uint64_t>(offset_or_soffset & 0x1fffffu) << 32;
  } else if (soffset_en) {
    word |= static_cast<std::uint64_t>(offset_or_soffset & 0x7fu) << 57;
  }
  return {static_cast<std::uint32_t>(word),
          static_cast<std::uint32_t>(word >> 32)};
}

std::array<std::uint32_t, 2> MakeSmemPrefetchPcRel(std::uint32_t op,
                                                   std::int32_t ioffset,
                                                   std::uint32_t soffset,
                                                   std::int32_t sdata) {
  std::uint64_t word = 0;
  word |= static_cast<std::uint64_t>(0x30u) << 26;
  word |= static_cast<std::uint64_t>(op & 0xffu) << 18;
  word |= static_cast<std::uint64_t>(static_cast<std::uint32_t>(sdata) & 0x1fu)
          << 6;
  word |= static_cast<std::uint64_t>(static_cast<std::uint32_t>(ioffset) &
                                     0x00ffffffu)
          << 32;
  word |= static_cast<std::uint64_t>(soffset & 0x7fu) << 57;
  return {static_cast<std::uint32_t>(word),
          static_cast<std::uint32_t>(word >> 32)};
}

std::array<std::uint32_t, 2> MakeSmemBasePrefetch(std::uint32_t op,
                                                  std::uint32_t sbase_start,
                                                  std::int32_t ioffset,
                                                  std::uint32_t soffset,
                                                  std::int32_t sdata) {
  std::uint64_t word = 0;
  word |= static_cast<std::uint64_t>(0x30u) << 26;
  word |= static_cast<std::uint64_t>(sbase_start >> 1);
  word |= static_cast<std::uint64_t>(static_cast<std::uint32_t>(sdata) & 0x1fu)
          << 6;
  word |= static_cast<std::uint64_t>(op & 0xffu) << 18;
  word |= static_cast<std::uint64_t>(static_cast<std::uint32_t>(ioffset) &
                                     0x00ffffffu)
          << 32;
  word |= static_cast<std::uint64_t>(soffset & 0x7fu) << 57;
  return {static_cast<std::uint32_t>(word),
          static_cast<std::uint32_t>(word >> 32)};
}

std::array<std::uint32_t, 2> MakeSmemBufferLoad(std::uint32_t op,
                                                std::uint32_t sdst,
                                                std::uint32_t sbase_start,
                                                std::int32_t ioffset,
                                                std::uint32_t soffset) {
  std::uint64_t word = 0;
  word |= static_cast<std::uint64_t>(0x30u) << 26;
  word |= static_cast<std::uint64_t>(sbase_start >> 1);
  word |= static_cast<std::uint64_t>(sdst) << 6;
  word |= static_cast<std::uint64_t>(op & 0xffu) << 18;
  word |= static_cast<std::uint64_t>(static_cast<std::uint32_t>(ioffset) &
                                     0x00ffffffu)
          << 32;
  word |= static_cast<std::uint64_t>(soffset & 0x7fu) << 57;
  return {static_cast<std::uint32_t>(word),
          static_cast<std::uint32_t>(word >> 32)};
}

std::array<std::uint32_t, 2> MakeDs(std::uint32_t op,
                                    std::uint32_t vdst,
                                    std::uint32_t addr,
                                    std::uint32_t data0,
                                    std::uint32_t data1,
                                    std::uint32_t offset0,
                                    std::uint32_t offset1 = 0,
                                    bool gds = false) {
  std::uint64_t word = 0;
  word |= static_cast<std::uint64_t>(offset0 & 0xffu) << 0;
  word |= static_cast<std::uint64_t>(offset1 & 0xffu) << 8;
  word |= static_cast<std::uint64_t>(gds ? 1u : 0u) << 16;
  word |= static_cast<std::uint64_t>(op & 0xffu) << 17;
  word |= static_cast<std::uint64_t>(0x36u) << 26;
  word |= static_cast<std::uint64_t>(addr & 0xffu) << 32;
  word |= static_cast<std::uint64_t>(data0 & 0xffu) << 40;
  word |= static_cast<std::uint64_t>(data1 & 0xffu) << 48;
  word |= static_cast<std::uint64_t>(vdst & 0xffu) << 56;
  return {static_cast<std::uint32_t>(word),
          static_cast<std::uint32_t>(word >> 32)};
}

std::array<std::uint32_t, 2> MakeGlobal(std::uint32_t op,
                                        std::uint32_t vdst,
                                        std::uint32_t addr,
                                        std::uint32_t data,
                                        std::uint32_t saddr,
                                        std::int32_t offset) {
  std::uint64_t word = 0;
  word |= static_cast<std::uint64_t>(55u) << 26;
  word |= static_cast<std::uint64_t>(static_cast<std::uint32_t>(offset) &
                                     0x1fffu) << 0;
  word |= static_cast<std::uint64_t>(2u) << 14;
  word |= static_cast<std::uint64_t>(op & 0x7fu) << 18;
  word |= static_cast<std::uint64_t>(addr & 0xffu) << 32;
  word |= static_cast<std::uint64_t>(data & 0xffu) << 40;
  word |= static_cast<std::uint64_t>(saddr & 0x7fu) << 48;
  word |= static_cast<std::uint64_t>(vdst & 0xffu) << 56;
  return {static_cast<std::uint32_t>(word),
          static_cast<std::uint32_t>(word >> 32)};
}

}  // namespace

int main() {
  using namespace mirage::sim::isa;

  const InstructionCatalogMetadata& metadata =
      GetGfx1201ImportedInstructionMetadata();
  if (!Expect(metadata.gfx_target == "gfx1201", "expected gfx1201 target") ||
      !Expect(metadata.architecture_name == "AMD RDNA 4",
              "expected AMD RDNA 4 architecture") ||
      !Expect(metadata.source_xml == "amdgpu_isa_rdna4.xml",
              "expected RDNA4 source xml") ||
      !Expect(metadata.instruction_count == 1264u,
              "expected imported instruction count") ||
      !Expect(metadata.encoding_count == 5062u,
              "expected imported encoding count")) {
    return 1;
  }

  const auto support_buckets = GetGfx1201SupportBucketSummaries();
  if (!Expect(support_buckets.size() == 5u,
              "expected five support buckets") ||
      !Expect(support_buckets.front().instruction_count == 363u,
              "expected transferable_full count") ||
      !Expect(support_buckets.back().instruction_count == 668u,
              "expected new_vs_gfx950 count")) {
    return 1;
  }

  Gfx1201BinaryDecoder decoder;
  if (!Expect(decoder.Phase0EncodingFocus().size() == 12u,
              "expected phase-0 encoding focus list") ||
      !Expect(decoder.Phase1EncodingFocus().size() == 8u,
              "expected phase-1 encoding focus list") ||
      !Expect(decoder.Phase0ComputeSeeds().size() == 12u,
              "expected phase-0 compute seed list") ||
      !Expect(decoder.Phase0ComputeSelectorRules().size() == 12u,
              "expected phase-0 selector rule list") ||
      !Expect(decoder.Phase0ExecutableOpcodes().size() == 459u,
              "expected phase-0 executable opcode slice") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_DCACHE_INV"),
              "expected S_DCACHE_INV executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_INV"),
              "expected GLOBAL_INV executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_WB"),
              "expected GLOBAL_WB executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_WBINV"),
              "expected GLOBAL_WBINV executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_LOAD_U8"),
              "expected GLOBAL_LOAD_U8 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_LOAD_I8"),
              "expected GLOBAL_LOAD_I8 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_LOAD_U16"),
              "expected GLOBAL_LOAD_U16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_LOAD_I16"),
              "expected GLOBAL_LOAD_I16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_LOAD_B32"),
              "expected GLOBAL_LOAD_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_LOAD_B64"),
              "expected GLOBAL_LOAD_B64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_LOAD_B96"),
              "expected GLOBAL_LOAD_B96 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_LOAD_B128"),
              "expected GLOBAL_LOAD_B128 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_LOAD_ADDTID_B32"),
              "expected GLOBAL_LOAD_ADDTID_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_LOAD_BLOCK"),
              "expected GLOBAL_LOAD_BLOCK executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_LOAD_TR_B64"),
              "expected GLOBAL_LOAD_TR_B64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_LOAD_TR_B128"),
              "expected GLOBAL_LOAD_TR_B128 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_LOAD_D16_U8"),
              "expected GLOBAL_LOAD_D16_U8 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_LOAD_D16_I8"),
              "expected GLOBAL_LOAD_D16_I8 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_LOAD_D16_B16"),
              "expected GLOBAL_LOAD_D16_B16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_LOAD_D16_HI_U8"),
              "expected GLOBAL_LOAD_D16_HI_U8 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_LOAD_D16_HI_I8"),
              "expected GLOBAL_LOAD_D16_HI_I8 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_LOAD_D16_HI_B16"),
              "expected GLOBAL_LOAD_D16_HI_B16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_STORE_B8"),
              "expected GLOBAL_STORE_B8 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_STORE_B16"),
              "expected GLOBAL_STORE_B16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_STORE_B32"),
              "expected GLOBAL_STORE_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_STORE_B64"),
              "expected GLOBAL_STORE_B64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_STORE_B96"),
              "expected GLOBAL_STORE_B96 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_STORE_B128"),
              "expected GLOBAL_STORE_B128 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_STORE_ADDTID_B32"),
              "expected GLOBAL_STORE_ADDTID_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_STORE_BLOCK"),
              "expected GLOBAL_STORE_BLOCK executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_STORE_D16_HI_B8"),
              "expected GLOBAL_STORE_D16_HI_B8 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_STORE_D16_HI_B16"),
              "expected GLOBAL_STORE_D16_HI_B16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_SWAP_B32"),
              "expected GLOBAL_ATOMIC_SWAP_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_CMPSWAP_B32"),
              "expected GLOBAL_ATOMIC_CMPSWAP_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_ADD_U32"),
              "expected GLOBAL_ATOMIC_ADD_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_SUB_U32"),
              "expected GLOBAL_ATOMIC_SUB_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_SUB_CLAMP_U32"),
              "expected GLOBAL_ATOMIC_SUB_CLAMP_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_MIN_I32"),
              "expected GLOBAL_ATOMIC_MIN_I32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_MIN_U32"),
              "expected GLOBAL_ATOMIC_MIN_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_MAX_I32"),
              "expected GLOBAL_ATOMIC_MAX_I32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_MAX_U32"),
              "expected GLOBAL_ATOMIC_MAX_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_AND_B32"),
              "expected GLOBAL_ATOMIC_AND_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_OR_B32"),
              "expected GLOBAL_ATOMIC_OR_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_XOR_B32"),
              "expected GLOBAL_ATOMIC_XOR_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_INC_U32"),
              "expected GLOBAL_ATOMIC_INC_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_DEC_U32"),
              "expected GLOBAL_ATOMIC_DEC_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_COND_SUB_U32"),
              "expected GLOBAL_ATOMIC_COND_SUB_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_SWAP_B64"),
              "expected GLOBAL_ATOMIC_SWAP_B64 executable decode support") ||
      !Expect(
          decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_CMPSWAP_B64"),
          "expected GLOBAL_ATOMIC_CMPSWAP_B64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_ADD_U64"),
              "expected GLOBAL_ATOMIC_ADD_U64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_SUB_U64"),
              "expected GLOBAL_ATOMIC_SUB_U64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_MIN_I64"),
              "expected GLOBAL_ATOMIC_MIN_I64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_MIN_U64"),
              "expected GLOBAL_ATOMIC_MIN_U64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_MAX_I64"),
              "expected GLOBAL_ATOMIC_MAX_I64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_MAX_U64"),
              "expected GLOBAL_ATOMIC_MAX_U64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_AND_B64"),
              "expected GLOBAL_ATOMIC_AND_B64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_OR_B64"),
              "expected GLOBAL_ATOMIC_OR_B64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_XOR_B64"),
              "expected GLOBAL_ATOMIC_XOR_B64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_INC_U64"),
              "expected GLOBAL_ATOMIC_INC_U64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_DEC_U64"),
              "expected GLOBAL_ATOMIC_DEC_U64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_ADD_F32"),
              "expected GLOBAL_ATOMIC_ADD_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_PK_ADD_F16"),
              "expected GLOBAL_ATOMIC_PK_ADD_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_PK_ADD_BF16"),
              "expected GLOBAL_ATOMIC_PK_ADD_BF16 executable decode support") ||
      !Expect(
          decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_MIN_NUM_F32"),
          "expected GLOBAL_ATOMIC_MIN_NUM_F32 executable decode support") ||
      !Expect(
          decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_MAX_NUM_F32"),
          "expected GLOBAL_ATOMIC_MAX_NUM_F32 executable decode support") ||
      !Expect(
          decoder.SupportsPhase0ExecutableOpcode("GLOBAL_ATOMIC_ORDERED_ADD_B64"),
          "expected GLOBAL_ATOMIC_ORDERED_ADD_B64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_NOP"),
              "expected DS_NOP executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_ADD_RTN_F32"),
              "expected DS_ADD_RTN_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_ADD_F32"),
              "expected DS_ADD_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_ADD_RTN_U32"),
              "expected DS_ADD_RTN_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_ADD_U32"),
              "expected DS_ADD_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_SUB_RTN_U32"),
              "expected DS_SUB_RTN_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_SUB_U32"),
              "expected DS_SUB_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_RSUB_RTN_U32"),
              "expected DS_RSUB_RTN_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_RSUB_U32"),
              "expected DS_RSUB_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_INC_RTN_U32"),
              "expected DS_INC_RTN_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_INC_U32"),
              "expected DS_INC_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_DEC_RTN_U32"),
              "expected DS_DEC_RTN_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_DEC_U32"),
              "expected DS_DEC_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_MIN_RTN_I32"),
              "expected DS_MIN_RTN_I32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_MIN_I32"),
              "expected DS_MIN_I32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_MIN_RTN_U32"),
              "expected DS_MIN_RTN_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_MIN_U32"),
              "expected DS_MIN_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_MAX_RTN_I32"),
              "expected DS_MAX_RTN_I32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_MAX_I32"),
              "expected DS_MAX_I32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_MAX_RTN_U32"),
              "expected DS_MAX_RTN_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_MAX_U32"),
              "expected DS_MAX_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_AND_RTN_B32"),
              "expected DS_AND_RTN_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_AND_B32"),
              "expected DS_AND_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_OR_RTN_B32"),
              "expected DS_OR_RTN_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_OR_B32"),
              "expected DS_OR_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_XOR_RTN_B32"),
              "expected DS_XOR_RTN_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_XOR_B32"),
              "expected DS_XOR_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_LOAD_B32"),
              "expected DS_LOAD_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_LOAD_B64"),
              "expected DS_LOAD_B64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_LOAD_B96"),
              "expected DS_LOAD_B96 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_LOAD_B128"),
              "expected DS_LOAD_B128 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_LOAD_I8"),
              "expected DS_LOAD_I8 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_LOAD_U8"),
              "expected DS_LOAD_U8 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_LOAD_I16"),
              "expected DS_LOAD_I16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_LOAD_U16"),
              "expected DS_LOAD_U16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_STORE_B8"),
              "expected DS_STORE_B8 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_STORE_B16"),
              "expected DS_STORE_B16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_STORE_B32"),
              "expected DS_STORE_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_STORE_B64"),
              "expected DS_STORE_B64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_STORE_B96"),
              "expected DS_STORE_B96 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("DS_STORE_B128"),
              "expected DS_STORE_B128 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_LOAD_B32"),
              "expected S_LOAD_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_LOAD_B64"),
              "expected S_LOAD_B64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_LOAD_B96"),
              "expected S_LOAD_B96 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_LOAD_B128"),
              "expected S_LOAD_B128 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_LOAD_B256"),
              "expected S_LOAD_B256 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_LOAD_B512"),
              "expected S_LOAD_B512 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_BUFFER_LOAD_B32"),
              "expected S_BUFFER_LOAD_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_BUFFER_LOAD_B64"),
              "expected S_BUFFER_LOAD_B64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_BUFFER_LOAD_B96"),
              "expected S_BUFFER_LOAD_B96 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_BUFFER_LOAD_B128"),
              "expected S_BUFFER_LOAD_B128 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_BUFFER_LOAD_B256"),
              "expected S_BUFFER_LOAD_B256 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_BUFFER_LOAD_B512"),
              "expected S_BUFFER_LOAD_B512 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_LOAD_I8"),
              "expected S_LOAD_I8 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_LOAD_U8"),
              "expected S_LOAD_U8 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_LOAD_I16"),
              "expected S_LOAD_I16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_LOAD_U16"),
              "expected S_LOAD_U16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_BUFFER_LOAD_I8"),
              "expected S_BUFFER_LOAD_I8 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_BUFFER_LOAD_U8"),
              "expected S_BUFFER_LOAD_U8 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_BUFFER_LOAD_I16"),
              "expected S_BUFFER_LOAD_I16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_BUFFER_LOAD_U16"),
              "expected S_BUFFER_LOAD_U16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_PREFETCH_INST"),
              "expected S_PREFETCH_INST executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_PREFETCH_INST_PC_REL"),
              "expected S_PREFETCH_INST_PC_REL executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_PREFETCH_DATA"),
              "expected S_PREFETCH_DATA executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_BUFFER_PREFETCH_DATA"),
              "expected S_BUFFER_PREFETCH_DATA executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_PREFETCH_DATA_PC_REL"),
              "expected S_PREFETCH_DATA_PC_REL executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_ATC_PROBE"),
              "expected S_ATC_PROBE executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_ATC_PROBE_BUFFER"),
              "expected S_ATC_PROBE_BUFFER executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_ADD_U32"),
              "expected S_ADD_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_SUB_U32"),
              "expected S_SUB_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_CMP_EQ_I32"),
              "expected S_CMP_EQ_I32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_CMP_EQ_U32"),
              "expected S_CMP_EQ_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_CMP_GE_I32"),
              "expected S_CMP_GE_I32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_CMP_LT_U32"),
              "expected S_CMP_LT_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_CBRANCH_SCC1"),
              "expected S_CBRANCH_SCC1 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_CBRANCH_VCCNZ"),
              "expected S_CBRANCH_VCCNZ executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_CBRANCH_EXECZ"),
              "expected S_CBRANCH_EXECZ executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_NOT_B32"),
              "expected V_NOT_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CLZ_I32_U32"),
              "expected V_CLZ_I32_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CLS_I32"),
              "expected V_CLS_I32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMP_EQ_U32"),
              "expected V_CMP_EQ_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMPX_EQ_U32"),
              "expected V_CMPX_EQ_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMP_EQ_I16"),
              "expected V_CMP_EQ_I16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMPX_EQ_U16"),
              "expected V_CMPX_EQ_U16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMP_EQ_F16"),
              "expected V_CMP_EQ_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMPX_CLASS_F16"),
              "expected V_CMPX_CLASS_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMP_EQ_F32"),
              "expected V_CMP_EQ_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMP_O_F32"),
              "expected V_CMP_O_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMPX_CLASS_F32"),
              "expected V_CMPX_CLASS_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMPX_U_F32"),
              "expected V_CMPX_U_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMP_EQ_F64"),
              "expected V_CMP_EQ_F64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMPX_CLASS_F64"),
              "expected V_CMPX_CLASS_F64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMP_EQ_I64"),
              "expected V_CMP_EQ_I64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMPX_EQ_U64"),
              "expected V_CMPX_EQ_U64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMP_GE_I32"),
              "expected V_CMP_GE_I32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMPX_GE_I32"),
              "expected V_CMPX_GE_I32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_F32_UBYTE3"),
              "expected V_CVT_F32_UBYTE3 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_READFIRSTLANE_B32"),
              "expected V_READFIRSTLANE_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MOVRELD_B32"),
              "expected V_MOVRELD_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MOVRELS_B32"),
              "expected V_MOVRELS_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MOVRELSD_B32"),
              "expected V_MOVRELSD_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MOVRELSD_2_B32"),
              "expected V_MOVRELSD_2_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SWAPREL_B32"),
              "expected V_SWAPREL_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_ADD_U32"),
              "expected V_ADD_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SUB_U32"),
              "expected V_SUB_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SUBREV_U32"),
              "expected V_SUBREV_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CNDMASK_B32"),
              "expected V_CNDMASK_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_XOR_B32"),
              "expected V_XOR_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_NOP"),
              "expected V_NOP executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_PIPEFLUSH"),
              "expected V_PIPEFLUSH executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MOV_B16"),
              "expected V_MOV_B16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_PERMLANE64_B32"),
              "expected V_PERMLANE64_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SWAP_B32"),
              "expected V_SWAP_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SWAP_B16"),
              "expected V_SWAP_B16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_NOT_B16"),
              "expected V_NOT_B16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_F32_I32"),
              "expected V_CVT_F32_I32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_F32_FP8"),
              "expected V_CVT_F32_FP8 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_F32_BF8"),
              "expected V_CVT_F32_BF8 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_PK_F32_FP8"),
              "expected V_CVT_PK_F32_FP8 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_PK_F32_BF8"),
              "expected V_CVT_PK_F32_BF8 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_NEAREST_I32_F32"),
              "expected V_CVT_NEAREST_I32_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_F16_F32"),
              "expected V_CVT_F16_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_F16_I16"),
              "expected V_CVT_F16_I16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_F16_U16"),
              "expected V_CVT_F16_U16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_F32_F16"),
              "expected V_CVT_F32_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_I16_F16"),
              "expected V_CVT_I16_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_U16_F16"),
              "expected V_CVT_U16_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SAT_PK_U8_I16"),
              "expected V_SAT_PK_U8_I16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_OFF_F32_I4"),
              "expected V_CVT_OFF_F32_I4 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_NORM_I16_F16"),
              "expected V_CVT_NORM_I16_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_NORM_U16_F16"),
              "expected V_CVT_NORM_U16_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_RCP_F16"),
              "expected V_RCP_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_RSQ_F16"),
              "expected V_RSQ_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SQRT_F16"),
              "expected V_SQRT_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_EXP_F16"),
              "expected V_EXP_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_LOG_F16"),
              "expected V_LOG_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SIN_F16"),
              "expected V_SIN_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_COS_F16"),
              "expected V_COS_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_FREXP_EXP_I16_F16"),
              "expected V_FREXP_EXP_I16_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_FREXP_MANT_F16"),
              "expected V_FREXP_MANT_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_FRACT_F16"),
              "expected V_FRACT_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_TRUNC_F16"),
              "expected V_TRUNC_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CEIL_F16"),
              "expected V_CEIL_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_RNDNE_F16"),
              "expected V_RNDNE_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_FLOOR_F16"),
              "expected V_FLOOR_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_ADD_F16"),
              "expected V_ADD_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SUB_F16"),
              "expected V_SUB_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SUBREV_F16"),
              "expected V_SUBREV_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MUL_F16"),
              "expected V_MUL_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_PK_RTZ_F16_F32"),
              "expected V_CVT_PK_RTZ_F16_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_LDEXP_F16"),
              "expected V_LDEXP_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MIN_NUM_F16"),
              "expected V_MIN_NUM_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MAX_NUM_F16"),
              "expected V_MAX_NUM_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_ADD_F32"),
              "expected V_ADD_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SUB_F32"),
              "expected V_SUB_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SUBREV_F32"),
              "expected V_SUBREV_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MUL_F32"),
              "expected V_MUL_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MIN_NUM_F32"),
              "expected V_MIN_NUM_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MAX_NUM_F32"),
              "expected V_MAX_NUM_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_ADD_F64"),
              "expected V_ADD_F64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MUL_F64"),
              "expected V_MUL_F64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MIN_NUM_F64"),
              "expected V_MIN_NUM_F64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MAX_NUM_F64"),
              "expected V_MAX_NUM_F64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_XNOR_B32"),
              "expected V_XNOR_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MUL_I32_I24"),
              "expected V_MUL_I32_I24 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MUL_HI_I32_I24"),
              "expected V_MUL_HI_I32_I24 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MUL_U32_U24"),
              "expected V_MUL_U32_U24 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MUL_HI_U32_U24"),
              "expected V_MUL_HI_U32_U24 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_LSHLREV_B64"),
              "expected V_LSHLREV_B64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_ADD_CO_CI_U32"),
              "expected V_ADD_CO_CI_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SUB_CO_CI_U32"),
              "expected V_SUB_CO_CI_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SUBREV_CO_CI_U32"),
              "expected V_SUBREV_CO_CI_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MUL_DX9_ZERO_F32"),
              "expected V_MUL_DX9_ZERO_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_FMAC_F32"),
              "expected V_FMAC_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_FMAC_F16"),
              "expected V_FMAC_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_FMAMK_F16"),
              "expected V_FMAMK_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_FMAAK_F16"),
              "expected V_FMAAK_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_PK_FMAC_F16"),
              "expected V_PK_FMAC_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_I32_I16"),
              "expected V_CVT_I32_I16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_U32_U16"),
              "expected V_CVT_U32_U16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_F64_F32"),
              "expected V_CVT_F64_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_I32_F64"),
              "expected V_CVT_I32_F64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_RCP_F32"),
              "expected V_RCP_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_LOG_F32"),
              "expected V_LOG_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SQRT_F64"),
              "expected V_SQRT_F64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_FREXP_EXP_I32_F32"),
              "expected V_FREXP_EXP_I32_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_FRACT_F64"),
              "expected V_FRACT_F64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_TRUNC_F32"),
              "expected V_TRUNC_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_FLOOR_F64"),
              "expected V_FLOOR_F64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_I32_F32"),
              "expected V_CVT_I32_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_MOV_B32"),
              "expected S_MOV_B32 executable decode support") ||
      !Expect(!decoder.SupportsPhase0ExecutableOpcode("V_DOT2_F32_F16"),
              "expected V_DOT2_F32_F16 to remain outside executable decode slice")) {
    return 1;
  }

  DecodedInstruction decoded_instruction;
  std::size_t words_consumed = 99;
  std::string error_message;
  const std::array<std::uint32_t, 1> route_only_words{0u};
  if (!Expect(!decoder.DecodeInstruction(route_only_words, &decoded_instruction,
                                         &words_consumed, &error_message),
              "decoder should still fail outside executable seed slice") ||
      !Expect(words_consumed == 0u, "expected no words consumed on route-only miss") ||
      !Expect(error_message.find("ENC_VOP2 opcode 0") != std::string::npos,
              "expected phase-0 route in error") ||
      !Expect(error_message.find("no matching seed entry") != std::string::npos,
              "expected seed-aware route miss in error")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> endpgm_words{MakeSopp(48u)};
  if (!Expect(decoder.DecodeInstruction(endpgm_words, &decoded_instruction,
                                        &words_consumed, &error_message),
              "expected S_ENDPGM decode success") ||
      !Expect(words_consumed == 1u, "expected one dword consumed") ||
      !Expect(decoded_instruction.opcode == "S_ENDPGM",
              "expected S_ENDPGM opcode") ||
      !Expect(decoded_instruction.operand_count == 0u,
              "expected S_ENDPGM nullary decode")) {
    return 1;
  }

  const auto dcache_inv_words = MakeSmem(33u, 0u, 0u, true, 0u);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(dcache_inv_words.data(),
                                                 dcache_inv_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected S_DCACHE_INV decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "S_DCACHE_INV",
              "expected S_DCACHE_INV opcode") ||
      !Expect(decoded_instruction.operand_count == 0u,
              "expected S_DCACHE_INV nullary decode")) {
    return 1;
  }

  const auto global_inv_words = MakeGlobal(43u, 0u, 0u, 0u, 0u, 0);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(global_inv_words.data(),
                                                 global_inv_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected GLOBAL_INV decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "GLOBAL_INV",
              "expected GLOBAL_INV opcode") ||
      !Expect(decoded_instruction.operand_count == 0u,
              "expected GLOBAL_INV nullary decode")) {
    return 1;
  }

  const auto global_load_b32_words = MakeGlobal(20u, 24u, 40u, 11u, 30u, 16);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(global_load_b32_words.data(),
                                                 global_load_b32_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected GLOBAL_LOAD_B32 decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "GLOBAL_LOAD_B32",
              "expected GLOBAL_LOAD_B32 opcode") ||
      !Expect(decoded_instruction.operand_count == 4u,
              "expected GLOBAL_LOAD_B32 quaternary decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[0].index == 24u,
              "expected GLOBAL_LOAD_B32 vector destination") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[1].index == 40u,
              "expected GLOBAL_LOAD_B32 vector address") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[2].index == 30u,
              "expected GLOBAL_LOAD_B32 scalar address") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[3].imm32 == 16u,
              "expected GLOBAL_LOAD_B32 inline offset")) {
    return 1;
  }

  const auto global_load_b128_words = MakeGlobal(23u, 28u, 44u, 9u, 32u, 28);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(global_load_b128_words.data(),
                                                 global_load_b128_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected GLOBAL_LOAD_B128 decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "GLOBAL_LOAD_B128",
              "expected GLOBAL_LOAD_B128 opcode") ||
      !Expect(decoded_instruction.operand_count == 4u,
              "expected GLOBAL_LOAD_B128 quaternary decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[0].index == 28u,
              "expected GLOBAL_LOAD_B128 vector destination") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[1].index == 44u,
              "expected GLOBAL_LOAD_B128 vector address") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[2].index == 32u,
              "expected GLOBAL_LOAD_B128 scalar address") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[3].imm32 == 28u,
              "expected GLOBAL_LOAD_B128 inline offset")) {
    return 1;
  }

  const auto global_load_d16_u8_words = MakeGlobal(30u, 34u, 46u, 0u, 20u, 12);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(global_load_d16_u8_words.data(),
                                                 global_load_d16_u8_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected GLOBAL_LOAD_D16_U8 decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "GLOBAL_LOAD_D16_U8",
              "expected GLOBAL_LOAD_D16_U8 opcode") ||
      !Expect(decoded_instruction.operand_count == 4u,
              "expected GLOBAL_LOAD_D16_U8 quaternary decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[0].index == 34u,
              "expected GLOBAL_LOAD_D16_U8 vector destination") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[1].index == 46u,
              "expected GLOBAL_LOAD_D16_U8 vector address") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[2].index == 20u,
              "expected GLOBAL_LOAD_D16_U8 scalar address") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[3].imm32 == 12u,
              "expected GLOBAL_LOAD_D16_U8 inline offset")) {
    return 1;
  }

  const auto global_load_d16_hi_b16_words =
      MakeGlobal(35u, 35u, 48u, 0u, 22u, 24);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_load_d16_hi_b16_words.data(),
                      global_load_d16_hi_b16_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected GLOBAL_LOAD_D16_HI_B16 decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "GLOBAL_LOAD_D16_HI_B16",
              "expected GLOBAL_LOAD_D16_HI_B16 opcode") ||
      !Expect(decoded_instruction.operand_count == 4u,
              "expected GLOBAL_LOAD_D16_HI_B16 quaternary decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[0].index == 35u,
              "expected GLOBAL_LOAD_D16_HI_B16 vector destination") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[1].index == 48u,
              "expected GLOBAL_LOAD_D16_HI_B16 vector address") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[2].index == 22u,
              "expected GLOBAL_LOAD_D16_HI_B16 scalar address") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[3].imm32 == 24u,
              "expected GLOBAL_LOAD_D16_HI_B16 inline offset")) {
    return 1;
  }

  const auto global_load_tr_b64_words = MakeGlobal(88u, 36u, 49u, 0u, 23u, 32);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(global_load_tr_b64_words.data(),
                                                 global_load_tr_b64_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected GLOBAL_LOAD_TR_B64 decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "GLOBAL_LOAD_TR_B64",
              "expected GLOBAL_LOAD_TR_B64 opcode") ||
      !Expect(decoded_instruction.operand_count == 4u,
              "expected GLOBAL_LOAD_TR_B64 quaternary decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[0].index == 36u,
              "expected GLOBAL_LOAD_TR_B64 vector destination") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[1].index == 49u,
              "expected GLOBAL_LOAD_TR_B64 vector address") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[2].index == 23u,
              "expected GLOBAL_LOAD_TR_B64 scalar address") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[3].imm32 == 32u,
              "expected GLOBAL_LOAD_TR_B64 inline offset")) {
    return 1;
  }

  const auto global_load_tr_b128_words =
      MakeGlobal(87u, 38u, 50u, 0u, 24u, 36);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(global_load_tr_b128_words.data(),
                                                 global_load_tr_b128_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected GLOBAL_LOAD_TR_B128 decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "GLOBAL_LOAD_TR_B128",
              "expected GLOBAL_LOAD_TR_B128 opcode") ||
      !Expect(decoded_instruction.operand_count == 4u,
              "expected GLOBAL_LOAD_TR_B128 quaternary decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[0].index == 38u,
              "expected GLOBAL_LOAD_TR_B128 vector destination") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[1].index == 50u,
              "expected GLOBAL_LOAD_TR_B128 vector address") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[2].index == 24u,
              "expected GLOBAL_LOAD_TR_B128 scalar address") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[3].imm32 == 36u,
              "expected GLOBAL_LOAD_TR_B128 inline offset")) {
    return 1;
  }

  const auto global_load_addtid_b32_words =
      MakeGlobal(40u, 39u, 0u, 0u, 26u, 0);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_load_addtid_b32_words.data(),
                      global_load_addtid_b32_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected GLOBAL_LOAD_ADDTID_B32 decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "GLOBAL_LOAD_ADDTID_B32",
              "expected GLOBAL_LOAD_ADDTID_B32 opcode") ||
      !Expect(decoded_instruction.operand_count == 3u,
              "expected GLOBAL_LOAD_ADDTID_B32 ternary decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[0].index == 39u,
              "expected GLOBAL_LOAD_ADDTID_B32 vector destination") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[1].index == 26u,
              "expected GLOBAL_LOAD_ADDTID_B32 scalar address")) {
    return 1;
  }

  const auto global_load_block_words =
      MakeGlobal(83u, 40u, 52u, 0u, 27u, 12);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(global_load_block_words.data(),
                                                 global_load_block_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected GLOBAL_LOAD_BLOCK decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "GLOBAL_LOAD_BLOCK",
              "expected GLOBAL_LOAD_BLOCK opcode") ||
      !Expect(decoded_instruction.operand_count == 5u,
              "expected GLOBAL_LOAD_BLOCK five-operand decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[0].index == 40u,
              "expected GLOBAL_LOAD_BLOCK vector destination base") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[1].index == 52u,
              "expected GLOBAL_LOAD_BLOCK vector address") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[2].index == 27u,
              "expected GLOBAL_LOAD_BLOCK scalar address") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[3].imm32 == 12u,
              "expected GLOBAL_LOAD_BLOCK inline offset") ||
      !Expect(decoded_instruction.operands[4].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[4].index == 124u,
              "expected GLOBAL_LOAD_BLOCK implicit M0 source")) {
    return 1;
  }

  const auto global_store_b32_words = MakeGlobal(26u, 0u, 44u, 18u, 30u, 16);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(global_store_b32_words.data(),
                                                 global_store_b32_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected GLOBAL_STORE_B32 decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "GLOBAL_STORE_B32",
              "expected GLOBAL_STORE_B32 opcode") ||
      !Expect(decoded_instruction.operand_count == 4u,
              "expected GLOBAL_STORE_B32 quaternary decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[0].index == 18u,
              "expected GLOBAL_STORE_B32 vector data") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[1].index == 44u,
              "expected GLOBAL_STORE_B32 vector address") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[2].index == 30u,
              "expected GLOBAL_STORE_B32 scalar address") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[3].imm32 == 16u,
              "expected GLOBAL_STORE_B32 inline offset")) {
    return 1;
  }

  const auto global_store_b128_words = MakeGlobal(29u, 0u, 48u, 22u, 32u, 28);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(global_store_b128_words.data(),
                                                 global_store_b128_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected GLOBAL_STORE_B128 decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "GLOBAL_STORE_B128",
              "expected GLOBAL_STORE_B128 opcode") ||
      !Expect(decoded_instruction.operand_count == 4u,
              "expected GLOBAL_STORE_B128 quaternary decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[0].index == 22u,
              "expected GLOBAL_STORE_B128 vector data") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[1].index == 48u,
              "expected GLOBAL_STORE_B128 vector address") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[2].index == 32u,
              "expected GLOBAL_STORE_B128 scalar address") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[3].imm32 == 28u,
              "expected GLOBAL_STORE_B128 inline offset")) {
    return 1;
  }

  const auto global_store_d16_hi_b16_words =
      MakeGlobal(37u, 0u, 50u, 24u, 34u, 20);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_store_d16_hi_b16_words.data(),
                      global_store_d16_hi_b16_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected GLOBAL_STORE_D16_HI_B16 decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "GLOBAL_STORE_D16_HI_B16",
              "expected GLOBAL_STORE_D16_HI_B16 opcode") ||
      !Expect(decoded_instruction.operand_count == 4u,
              "expected GLOBAL_STORE_D16_HI_B16 quaternary decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[0].index == 24u,
              "expected GLOBAL_STORE_D16_HI_B16 vector data") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[1].index == 50u,
              "expected GLOBAL_STORE_D16_HI_B16 vector address") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[2].index == 34u,
              "expected GLOBAL_STORE_D16_HI_B16 scalar address") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[3].imm32 == 20u,
              "expected GLOBAL_STORE_D16_HI_B16 inline offset")) {
    return 1;
  }

  const auto global_store_addtid_b32_words =
      MakeGlobal(41u, 0u, 0u, 27u, 36u, 0);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_store_addtid_b32_words.data(),
                      global_store_addtid_b32_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected GLOBAL_STORE_ADDTID_B32 decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "GLOBAL_STORE_ADDTID_B32",
              "expected GLOBAL_STORE_ADDTID_B32 opcode") ||
      !Expect(decoded_instruction.operand_count == 3u,
              "expected GLOBAL_STORE_ADDTID_B32 ternary decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[0].index == 27u,
              "expected GLOBAL_STORE_ADDTID_B32 vector data") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[1].index == 36u,
              "expected GLOBAL_STORE_ADDTID_B32 scalar address")) {
    return 1;
  }

  const auto global_store_block_words =
      MakeGlobal(84u, 0u, 53u, 44u, 37u, 20);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(global_store_block_words.data(),
                                                 global_store_block_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected GLOBAL_STORE_BLOCK decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "GLOBAL_STORE_BLOCK",
              "expected GLOBAL_STORE_BLOCK opcode") ||
      !Expect(decoded_instruction.operand_count == 5u,
              "expected GLOBAL_STORE_BLOCK five-operand decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[0].index == 44u,
              "expected GLOBAL_STORE_BLOCK vector data base") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[1].index == 53u,
              "expected GLOBAL_STORE_BLOCK vector address") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[2].index == 37u,
              "expected GLOBAL_STORE_BLOCK scalar address") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[3].imm32 == 20u,
              "expected GLOBAL_STORE_BLOCK inline offset") ||
      !Expect(decoded_instruction.operands[4].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[4].index == 124u,
              "expected GLOBAL_STORE_BLOCK implicit M0 source")) {
    return 1;
  }

  const auto global_atomic_swap_b32_words =
      MakeGlobal(51u, 54u, 55u, 28u, 38u, 24);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_atomic_swap_b32_words.data(),
                      global_atomic_swap_b32_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected GLOBAL_ATOMIC_SWAP_B32 decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "GLOBAL_ATOMIC_SWAP_B32",
              "expected GLOBAL_ATOMIC_SWAP_B32 opcode") ||
      !Expect(decoded_instruction.operand_count == 5u,
              "expected GLOBAL_ATOMIC_SWAP_B32 five-operand decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[0].index == 54u,
              "expected GLOBAL_ATOMIC_SWAP_B32 vector destination") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[1].index == 28u,
              "expected GLOBAL_ATOMIC_SWAP_B32 vector data") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[2].index == 55u,
              "expected GLOBAL_ATOMIC_SWAP_B32 vector address") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[3].index == 38u,
              "expected GLOBAL_ATOMIC_SWAP_B32 scalar address") ||
      !Expect(decoded_instruction.operands[4].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[4].imm32 == 24u,
              "expected GLOBAL_ATOMIC_SWAP_B32 inline offset")) {
    return 1;
  }

  const auto global_atomic_cmpswap_b32_words =
      MakeGlobal(52u, 56u, 57u, 29u, 39u, 28);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_atomic_cmpswap_b32_words.data(),
                      global_atomic_cmpswap_b32_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected GLOBAL_ATOMIC_CMPSWAP_B32 decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "GLOBAL_ATOMIC_CMPSWAP_B32",
              "expected GLOBAL_ATOMIC_CMPSWAP_B32 opcode") ||
      !Expect(decoded_instruction.operand_count == 5u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B32 five-operand decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[0].index == 56u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B32 vector destination") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[1].index == 29u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B32 vector data pair base") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[2].index == 57u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B32 vector address") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[3].index == 39u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B32 scalar address") ||
      !Expect(decoded_instruction.operands[4].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[4].imm32 == 28u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B32 inline offset")) {
    return 1;
  }

  const auto global_atomic_swap_b64_words =
      MakeGlobal(65u, 62u, 63u, 40u, 40u, 32);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_atomic_swap_b64_words.data(),
                      global_atomic_swap_b64_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected GLOBAL_ATOMIC_SWAP_B64 decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "GLOBAL_ATOMIC_SWAP_B64",
              "expected GLOBAL_ATOMIC_SWAP_B64 opcode") ||
      !Expect(decoded_instruction.operand_count == 5u,
              "expected GLOBAL_ATOMIC_SWAP_B64 five-operand decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[0].index == 62u,
              "expected GLOBAL_ATOMIC_SWAP_B64 vector destination pair") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[1].index == 40u,
              "expected GLOBAL_ATOMIC_SWAP_B64 vector data pair") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[2].index == 63u,
              "expected GLOBAL_ATOMIC_SWAP_B64 vector address") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[3].index == 40u,
              "expected GLOBAL_ATOMIC_SWAP_B64 scalar address") ||
      !Expect(decoded_instruction.operands[4].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[4].imm32 == 32u,
              "expected GLOBAL_ATOMIC_SWAP_B64 inline offset")) {
    return 1;
  }

  const auto global_atomic_cmpswap_b64_words =
      MakeGlobal(66u, 64u, 65u, 41u, 41u, 36);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_atomic_cmpswap_b64_words.data(),
                      global_atomic_cmpswap_b64_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected GLOBAL_ATOMIC_CMPSWAP_B64 decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "GLOBAL_ATOMIC_CMPSWAP_B64",
              "expected GLOBAL_ATOMIC_CMPSWAP_B64 opcode") ||
      !Expect(decoded_instruction.operand_count == 5u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B64 five-operand decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[0].index == 64u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B64 vector destination pair") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[1].index == 41u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B64 vector data quad base") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[2].index == 65u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B64 vector address") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[3].index == 41u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B64 scalar address") ||
      !Expect(decoded_instruction.operands[4].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[4].imm32 == 36u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B64 inline offset")) {
    return 1;
  }

  const auto global_atomic_add_f32_words =
      MakeGlobal(86u, 66u, 67u, 42u, 42u, 40);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_atomic_add_f32_words.data(),
                      global_atomic_add_f32_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected GLOBAL_ATOMIC_ADD_F32 decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "GLOBAL_ATOMIC_ADD_F32",
              "expected GLOBAL_ATOMIC_ADD_F32 opcode") ||
      !Expect(decoded_instruction.operand_count == 5u,
              "expected GLOBAL_ATOMIC_ADD_F32 five-operand decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[0].index == 66u,
              "expected GLOBAL_ATOMIC_ADD_F32 vector destination") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[1].index == 42u,
              "expected GLOBAL_ATOMIC_ADD_F32 vector data") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[2].index == 67u,
              "expected GLOBAL_ATOMIC_ADD_F32 vector address") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[3].index == 42u,
              "expected GLOBAL_ATOMIC_ADD_F32 scalar address") ||
      !Expect(decoded_instruction.operands[4].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[4].imm32 == 40u,
              "expected GLOBAL_ATOMIC_ADD_F32 inline offset")) {
    return 1;
  }

  const auto global_atomic_pk_add_f16_words =
      MakeGlobal(89u, 68u, 69u, 43u, 43u, 44);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_atomic_pk_add_f16_words.data(),
                      global_atomic_pk_add_f16_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected GLOBAL_ATOMIC_PK_ADD_F16 decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "GLOBAL_ATOMIC_PK_ADD_F16",
              "expected GLOBAL_ATOMIC_PK_ADD_F16 opcode") ||
      !Expect(decoded_instruction.operand_count == 5u,
              "expected GLOBAL_ATOMIC_PK_ADD_F16 five-operand decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[0].index == 68u,
              "expected GLOBAL_ATOMIC_PK_ADD_F16 vector destination") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[1].index == 43u,
              "expected GLOBAL_ATOMIC_PK_ADD_F16 vector data") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[2].index == 69u,
              "expected GLOBAL_ATOMIC_PK_ADD_F16 vector address") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[3].index == 43u,
              "expected GLOBAL_ATOMIC_PK_ADD_F16 scalar address") ||
      !Expect(decoded_instruction.operands[4].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[4].imm32 == 44u,
              "expected GLOBAL_ATOMIC_PK_ADD_F16 inline offset")) {
    return 1;
  }

  const auto global_atomic_ordered_add_b64_words =
      MakeGlobal(115u, 70u, 71u, 44u, 44u, 48);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_atomic_ordered_add_b64_words.data(),
                      global_atomic_ordered_add_b64_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected GLOBAL_ATOMIC_ORDERED_ADD_B64 decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "GLOBAL_ATOMIC_ORDERED_ADD_B64",
              "expected GLOBAL_ATOMIC_ORDERED_ADD_B64 opcode") ||
      !Expect(decoded_instruction.operand_count == 5u,
              "expected GLOBAL_ATOMIC_ORDERED_ADD_B64 five-operand decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[0].index == 70u,
              "expected GLOBAL_ATOMIC_ORDERED_ADD_B64 vector destination") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[1].index == 44u,
              "expected GLOBAL_ATOMIC_ORDERED_ADD_B64 vector data") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[2].index == 71u,
              "expected GLOBAL_ATOMIC_ORDERED_ADD_B64 vector address") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[3].index == 44u,
              "expected GLOBAL_ATOMIC_ORDERED_ADD_B64 scalar address") ||
      !Expect(decoded_instruction.operands[4].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[4].imm32 == 48u,
              "expected GLOBAL_ATOMIC_ORDERED_ADD_B64 inline offset")) {
    return 1;
  }

  const auto ds_nop_words = MakeDs(20u, 0u, 0u, 0u, 0u, 0u);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(ds_nop_words.data(),
                                                 ds_nop_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected DS_NOP decode success") ||
      !Expect(words_consumed == 2u, "expected DS_NOP two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "DS_NOP",
              "expected DS_NOP opcode") ||
      !Expect(decoded_instruction.operand_count == 0u,
              "expected DS_NOP nullary decode")) {
    return 1;
  }

  const auto ds_add_u32_words = MakeDs(0u, 99u, 60u, 61u, 62u, 0x56u, 0x34u);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(ds_add_u32_words.data(),
                                                 ds_add_u32_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected DS_ADD_U32 decode success") ||
      !Expect(words_consumed == 2u, "expected DS_ADD_U32 two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "DS_ADD_U32",
              "expected DS_ADD_U32 opcode") ||
      !Expect(decoded_instruction.operand_count == 4u,
              "expected DS_ADD_U32 four-operand decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[0].index == 60u,
              "expected DS_ADD_U32 address VGPR") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[1].index == 61u,
              "expected DS_ADD_U32 data VGPR") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[2].imm32 == 0x56u,
              "expected DS_ADD_U32 offset0") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[3].imm32 == 0x34u,
              "expected DS_ADD_U32 offset1")) {
    return 1;
  }

  const auto ds_add_rtn_u32_words =
      MakeDs(32u, 59u, 60u, 61u, 62u, 0x56u, 0x34u);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(ds_add_rtn_u32_words.data(),
                                                 ds_add_rtn_u32_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected DS_ADD_RTN_U32 decode success") ||
      !Expect(words_consumed == 2u,
              "expected DS_ADD_RTN_U32 two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "DS_ADD_RTN_U32",
              "expected DS_ADD_RTN_U32 opcode") ||
      !Expect(decoded_instruction.operand_count == 5u,
              "expected DS_ADD_RTN_U32 five-operand decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[0].index == 59u,
              "expected DS_ADD_RTN_U32 destination VGPR") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[1].index == 60u,
              "expected DS_ADD_RTN_U32 address VGPR") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[2].index == 61u,
              "expected DS_ADD_RTN_U32 data VGPR") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[3].imm32 == 0x56u,
              "expected DS_ADD_RTN_U32 offset0") ||
      !Expect(decoded_instruction.operands[4].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[4].imm32 == 0x34u,
              "expected DS_ADD_RTN_U32 offset1")) {
    return 1;
  }

  const auto ds_load_b128_words = MakeDs(255u, 59u, 60u, 61u, 62u, 0x56u);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(ds_load_b128_words.data(),
                                                 ds_load_b128_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected DS_LOAD_B128 decode success") ||
      !Expect(words_consumed == 2u,
              "expected DS_LOAD_B128 two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "DS_LOAD_B128",
              "expected DS_LOAD_B128 opcode") ||
      !Expect(decoded_instruction.operand_count == 3u,
              "expected DS_LOAD_B128 three-operand decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[0].index == 59u,
              "expected DS_LOAD_B128 destination VGPR") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[1].index == 60u,
              "expected DS_LOAD_B128 address VGPR") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[2].imm32 == 0x56u,
              "expected DS_LOAD_B128 offset0")) {
    return 1;
  }

  const auto ds_store_b128_words = MakeDs(223u, 59u, 60u, 61u, 62u, 0x34u);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(ds_store_b128_words.data(),
                                                 ds_store_b128_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected DS_STORE_B128 decode success") ||
      !Expect(words_consumed == 2u,
              "expected DS_STORE_B128 two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "DS_STORE_B128",
              "expected DS_STORE_B128 opcode") ||
      !Expect(decoded_instruction.operand_count == 3u,
              "expected DS_STORE_B128 three-operand decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[0].index == 61u,
              "expected DS_STORE_B128 data VGPR") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kVgpr &&
                  decoded_instruction.operands[1].index == 60u,
              "expected DS_STORE_B128 address VGPR") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[2].imm32 == 0x34u,
              "expected DS_STORE_B128 offset0")) {
    return 1;
  }

  const auto load_b32_words = MakeSmem(0u, 18u, 4u, true, 12u);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(load_b32_words.data(),
                                                 load_b32_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected S_LOAD_B32 decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "S_LOAD_B32",
              "expected S_LOAD_B32 opcode") ||
      !Expect(decoded_instruction.operand_count == 3u,
              "expected S_LOAD_B32 ternary decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[0].index == 18u,
              "expected S_LOAD_B32 scalar destination") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[1].index == 4u,
              "expected S_LOAD_B32 64-bit scalar base") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[2].imm32 == 12u,
              "expected S_LOAD_B32 inline offset")) {
    return 1;
  }

  const auto load_b64_words = MakeSmem(1u, 20u, 8u, false, 31u, true);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(load_b64_words.data(),
                                                 load_b64_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected S_LOAD_B64 decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "S_LOAD_B64",
              "expected S_LOAD_B64 opcode") ||
      !Expect(decoded_instruction.operand_count == 3u,
              "expected S_LOAD_B64 ternary decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[0].index == 20u,
              "expected S_LOAD_B64 scalar destination") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[1].index == 8u,
              "expected S_LOAD_B64 64-bit scalar base") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[2].index == 31u,
              "expected S_LOAD_B64 scalar offset register")) {
    return 1;
  }

  const auto load_b96_words = MakeSmem(5u, 22u, 10u, true, 16u);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(load_b96_words.data(),
                                                 load_b96_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected S_LOAD_B96 decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "S_LOAD_B96",
              "expected S_LOAD_B96 opcode") ||
      !Expect(decoded_instruction.operand_count == 3u,
              "expected S_LOAD_B96 ternary decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[0].index == 22u,
              "expected S_LOAD_B96 scalar destination") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[1].index == 10u,
              "expected S_LOAD_B96 64-bit scalar base") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[2].imm32 == 16u,
              "expected S_LOAD_B96 inline offset")) {
    return 1;
  }

  const auto load_b128_words = MakeSmem(2u, 26u, 12u, false, 29u, true);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(load_b128_words.data(),
                                                 load_b128_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected S_LOAD_B128 decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "S_LOAD_B128",
              "expected S_LOAD_B128 opcode") ||
      !Expect(decoded_instruction.operand_count == 3u,
              "expected S_LOAD_B128 ternary decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[0].index == 26u,
              "expected S_LOAD_B128 scalar destination") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[1].index == 12u,
              "expected S_LOAD_B128 64-bit scalar base") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[2].index == 29u,
              "expected S_LOAD_B128 scalar offset register")) {
    return 1;
  }

  const auto buffer_load_b32_words = MakeSmemBufferLoad(16u, 30u, 14u, 24, 19u);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(buffer_load_b32_words.data(),
                                                 buffer_load_b32_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected S_BUFFER_LOAD_B32 decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "S_BUFFER_LOAD_B32",
              "expected S_BUFFER_LOAD_B32 opcode") ||
      !Expect(decoded_instruction.operand_count == 4u,
              "expected S_BUFFER_LOAD_B32 quaternary decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[0].index == 30u,
              "expected S_BUFFER_LOAD_B32 scalar destination") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[1].index == 14u,
              "expected S_BUFFER_LOAD_B32 resource base") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[2].imm32 == 24u,
              "expected S_BUFFER_LOAD_B32 inline ioffset") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[3].index == 19u,
              "expected S_BUFFER_LOAD_B32 scalar soffset")) {
    return 1;
  }

  const auto buffer_load_b128_words =
      MakeSmemBufferLoad(18u, 34u, 16u, -12, 23u);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(buffer_load_b128_words.data(),
                                                 buffer_load_b128_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected S_BUFFER_LOAD_B128 decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "S_BUFFER_LOAD_B128",
              "expected S_BUFFER_LOAD_B128 opcode") ||
      !Expect(decoded_instruction.operand_count == 4u,
              "expected S_BUFFER_LOAD_B128 quaternary decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[0].index == 34u,
              "expected S_BUFFER_LOAD_B128 scalar destination") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[1].index == 16u,
              "expected S_BUFFER_LOAD_B128 resource base") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kImm32 &&
                  static_cast<std::int32_t>(decoded_instruction.operands[2].imm32) ==
                      -12,
              "expected S_BUFFER_LOAD_B128 inline ioffset") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[3].index == 23u,
              "expected S_BUFFER_LOAD_B128 scalar soffset")) {
    return 1;
  }

  const auto prefetch_inst_pc_rel_words =
      MakeSmemPrefetchPcRel(37u, -32, 9u, -3);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(prefetch_inst_pc_rel_words.data(),
                                                 prefetch_inst_pc_rel_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected S_PREFETCH_INST_PC_REL decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "S_PREFETCH_INST_PC_REL",
              "expected S_PREFETCH_INST_PC_REL opcode") ||
      !Expect(decoded_instruction.operand_count == 3u,
              "expected S_PREFETCH_INST_PC_REL ternary decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[0].imm32 ==
                      static_cast<std::uint32_t>(-32),
              "expected sign-extended S_PREFETCH_INST_PC_REL ioffset") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[1].index == 9u,
              "expected S_PREFETCH_INST_PC_REL soffset register") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[2].imm32 ==
                      static_cast<std::uint32_t>(-3),
              "expected sign-extended S_PREFETCH_INST_PC_REL sdata")) {
    return 1;
  }

  const auto prefetch_inst_words = MakeSmemBasePrefetch(36u, 8u, -16, 11u, -4);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(prefetch_inst_words.data(),
                                                 prefetch_inst_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected S_PREFETCH_INST decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "S_PREFETCH_INST",
              "expected S_PREFETCH_INST opcode") ||
      !Expect(decoded_instruction.operand_count == 4u,
              "expected S_PREFETCH_INST four-operand decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[0].index == 8u,
              "expected S_PREFETCH_INST 64-bit scalar base") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[1].imm32 ==
                      static_cast<std::uint32_t>(-16),
              "expected sign-extended S_PREFETCH_INST ioffset") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[2].index == 11u,
              "expected S_PREFETCH_INST scalar offset register") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[3].imm32 ==
                      static_cast<std::uint32_t>(-4),
              "expected sign-extended S_PREFETCH_INST sdata")) {
    return 1;
  }

  const auto prefetch_data_words = MakeSmemBasePrefetch(38u, 12u, 64, 7u, 3);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(prefetch_data_words.data(),
                                                 prefetch_data_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected S_PREFETCH_DATA decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "S_PREFETCH_DATA",
              "expected S_PREFETCH_DATA opcode") ||
      !Expect(decoded_instruction.operand_count == 4u,
              "expected S_PREFETCH_DATA four-operand decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[0].index == 12u,
              "expected S_PREFETCH_DATA 64-bit scalar base") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[1].imm32 == 64u,
              "expected S_PREFETCH_DATA ioffset") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[2].index == 7u,
              "expected S_PREFETCH_DATA scalar offset register") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[3].imm32 == 3u,
              "expected S_PREFETCH_DATA sdata")) {
    return 1;
  }

  const auto buffer_prefetch_words =
      MakeSmemBasePrefetch(39u, 20u, 24, 13u, -1);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(buffer_prefetch_words.data(),
                                                 buffer_prefetch_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected S_BUFFER_PREFETCH_DATA decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "S_BUFFER_PREFETCH_DATA",
              "expected S_BUFFER_PREFETCH_DATA opcode") ||
      !Expect(decoded_instruction.operand_count == 4u,
              "expected S_BUFFER_PREFETCH_DATA four-operand decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[0].index == 20u,
              "expected S_BUFFER_PREFETCH_DATA 128-bit scalar base") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[1].imm32 == 24u,
              "expected S_BUFFER_PREFETCH_DATA ioffset") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[2].index == 13u,
              "expected S_BUFFER_PREFETCH_DATA scalar offset register") ||
      !Expect(decoded_instruction.operands[3].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[3].imm32 ==
                      static_cast<std::uint32_t>(-1),
              "expected sign-extended S_BUFFER_PREFETCH_DATA sdata")) {
    return 1;
  }

  const auto atc_probe_words = MakeSmem(34u, 42u, 6u, false, 17u, true);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(atc_probe_words.data(),
                                                 atc_probe_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected S_ATC_PROBE decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "S_ATC_PROBE",
              "expected S_ATC_PROBE opcode") ||
      !Expect(decoded_instruction.operand_count == 3u,
              "expected S_ATC_PROBE ternary decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[0].imm32 == 42u,
              "expected S_ATC_PROBE immediate sdata") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[1].index == 6u,
              "expected S_ATC_PROBE 64-bit scalar base") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[2].index == 17u,
              "expected S_ATC_PROBE scalar offset register")) {
    return 1;
  }

  const auto atc_probe_buffer_words = MakeSmem(35u, 55u, 10u, true, 0x1abcdu);
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(atc_probe_buffer_words.data(),
                                                 atc_probe_buffer_words.size()),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected S_ATC_PROBE_BUFFER decode success") ||
      !Expect(words_consumed == 2u, "expected two dwords consumed") ||
      !Expect(decoded_instruction.opcode == "S_ATC_PROBE_BUFFER",
              "expected S_ATC_PROBE_BUFFER opcode") ||
      !Expect(decoded_instruction.operand_count == 3u,
              "expected S_ATC_PROBE_BUFFER ternary decode") ||
      !Expect(decoded_instruction.operands[0].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[0].imm32 == 55u,
              "expected S_ATC_PROBE_BUFFER immediate sdata") ||
      !Expect(decoded_instruction.operands[1].kind == OperandKind::kSgpr &&
                  decoded_instruction.operands[1].index == 10u,
              "expected S_ATC_PROBE_BUFFER 128-bit scalar base") ||
      !Expect(decoded_instruction.operands[2].kind == OperandKind::kImm32 &&
                  decoded_instruction.operands[2].imm32 == 0x1abcdu,
              "expected S_ATC_PROBE_BUFFER inline offset")) {
    return 1;
  }

  Gfx1201Interpreter interpreter;
  if (!Expect(interpreter.ExecutableSeedOpcodes().size() == 459u,
              "expected executable seed opcode list") ||
      !Expect(interpreter.Supports("S_ENDPGM"),
              "expected interpreter support for S_ENDPGM") ||
      !Expect(interpreter.Supports("S_DCACHE_INV"),
              "expected interpreter support for S_DCACHE_INV") ||
      !Expect(interpreter.Supports("GLOBAL_INV"),
              "expected interpreter support for GLOBAL_INV") ||
      !Expect(interpreter.Supports("GLOBAL_WB"),
              "expected interpreter support for GLOBAL_WB") ||
      !Expect(interpreter.Supports("GLOBAL_WBINV"),
              "expected interpreter support for GLOBAL_WBINV") ||
      !Expect(interpreter.Supports("GLOBAL_LOAD_U8"),
              "expected interpreter support for GLOBAL_LOAD_U8") ||
      !Expect(interpreter.Supports("GLOBAL_LOAD_I8"),
              "expected interpreter support for GLOBAL_LOAD_I8") ||
      !Expect(interpreter.Supports("GLOBAL_LOAD_U16"),
              "expected interpreter support for GLOBAL_LOAD_U16") ||
      !Expect(interpreter.Supports("GLOBAL_LOAD_I16"),
              "expected interpreter support for GLOBAL_LOAD_I16") ||
      !Expect(interpreter.Supports("GLOBAL_LOAD_B32"),
              "expected interpreter support for GLOBAL_LOAD_B32") ||
      !Expect(interpreter.Supports("GLOBAL_LOAD_B64"),
              "expected interpreter support for GLOBAL_LOAD_B64") ||
      !Expect(interpreter.Supports("GLOBAL_LOAD_B96"),
              "expected interpreter support for GLOBAL_LOAD_B96") ||
      !Expect(interpreter.Supports("GLOBAL_LOAD_B128"),
              "expected interpreter support for GLOBAL_LOAD_B128") ||
      !Expect(interpreter.Supports("GLOBAL_LOAD_ADDTID_B32"),
              "expected interpreter support for GLOBAL_LOAD_ADDTID_B32") ||
      !Expect(interpreter.Supports("GLOBAL_LOAD_BLOCK"),
              "expected interpreter support for GLOBAL_LOAD_BLOCK") ||
      !Expect(interpreter.Supports("GLOBAL_LOAD_TR_B64"),
              "expected interpreter support for GLOBAL_LOAD_TR_B64") ||
      !Expect(interpreter.Supports("GLOBAL_LOAD_TR_B128"),
              "expected interpreter support for GLOBAL_LOAD_TR_B128") ||
      !Expect(interpreter.Supports("GLOBAL_LOAD_D16_U8"),
              "expected interpreter support for GLOBAL_LOAD_D16_U8") ||
      !Expect(interpreter.Supports("GLOBAL_LOAD_D16_I8"),
              "expected interpreter support for GLOBAL_LOAD_D16_I8") ||
      !Expect(interpreter.Supports("GLOBAL_LOAD_D16_B16"),
              "expected interpreter support for GLOBAL_LOAD_D16_B16") ||
      !Expect(interpreter.Supports("GLOBAL_LOAD_D16_HI_U8"),
              "expected interpreter support for GLOBAL_LOAD_D16_HI_U8") ||
      !Expect(interpreter.Supports("GLOBAL_LOAD_D16_HI_I8"),
              "expected interpreter support for GLOBAL_LOAD_D16_HI_I8") ||
      !Expect(interpreter.Supports("GLOBAL_LOAD_D16_HI_B16"),
              "expected interpreter support for GLOBAL_LOAD_D16_HI_B16") ||
      !Expect(interpreter.Supports("GLOBAL_STORE_B8"),
              "expected interpreter support for GLOBAL_STORE_B8") ||
      !Expect(interpreter.Supports("GLOBAL_STORE_B16"),
              "expected interpreter support for GLOBAL_STORE_B16") ||
      !Expect(interpreter.Supports("GLOBAL_STORE_B32"),
              "expected interpreter support for GLOBAL_STORE_B32") ||
      !Expect(interpreter.Supports("GLOBAL_STORE_B64"),
              "expected interpreter support for GLOBAL_STORE_B64") ||
      !Expect(interpreter.Supports("GLOBAL_STORE_B96"),
              "expected interpreter support for GLOBAL_STORE_B96") ||
      !Expect(interpreter.Supports("GLOBAL_STORE_B128"),
              "expected interpreter support for GLOBAL_STORE_B128") ||
      !Expect(interpreter.Supports("GLOBAL_STORE_ADDTID_B32"),
              "expected interpreter support for GLOBAL_STORE_ADDTID_B32") ||
      !Expect(interpreter.Supports("GLOBAL_STORE_BLOCK"),
              "expected interpreter support for GLOBAL_STORE_BLOCK") ||
      !Expect(interpreter.Supports("GLOBAL_STORE_D16_HI_B8"),
              "expected interpreter support for GLOBAL_STORE_D16_HI_B8") ||
      !Expect(interpreter.Supports("GLOBAL_STORE_D16_HI_B16"),
              "expected interpreter support for GLOBAL_STORE_D16_HI_B16") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_SWAP_B32"),
              "expected interpreter support for GLOBAL_ATOMIC_SWAP_B32") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_CMPSWAP_B32"),
              "expected interpreter support for GLOBAL_ATOMIC_CMPSWAP_B32") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_ADD_U32"),
              "expected interpreter support for GLOBAL_ATOMIC_ADD_U32") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_SUB_U32"),
              "expected interpreter support for GLOBAL_ATOMIC_SUB_U32") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_SUB_CLAMP_U32"),
              "expected interpreter support for GLOBAL_ATOMIC_SUB_CLAMP_U32") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_MIN_I32"),
              "expected interpreter support for GLOBAL_ATOMIC_MIN_I32") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_MIN_U32"),
              "expected interpreter support for GLOBAL_ATOMIC_MIN_U32") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_MAX_I32"),
              "expected interpreter support for GLOBAL_ATOMIC_MAX_I32") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_MAX_U32"),
              "expected interpreter support for GLOBAL_ATOMIC_MAX_U32") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_AND_B32"),
              "expected interpreter support for GLOBAL_ATOMIC_AND_B32") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_OR_B32"),
              "expected interpreter support for GLOBAL_ATOMIC_OR_B32") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_XOR_B32"),
              "expected interpreter support for GLOBAL_ATOMIC_XOR_B32") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_INC_U32"),
              "expected interpreter support for GLOBAL_ATOMIC_INC_U32") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_DEC_U32"),
              "expected interpreter support for GLOBAL_ATOMIC_DEC_U32") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_COND_SUB_U32"),
              "expected interpreter support for GLOBAL_ATOMIC_COND_SUB_U32") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_SWAP_B64"),
              "expected interpreter support for GLOBAL_ATOMIC_SWAP_B64") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_CMPSWAP_B64"),
              "expected interpreter support for GLOBAL_ATOMIC_CMPSWAP_B64") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_ADD_U64"),
              "expected interpreter support for GLOBAL_ATOMIC_ADD_U64") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_SUB_U64"),
              "expected interpreter support for GLOBAL_ATOMIC_SUB_U64") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_MIN_I64"),
              "expected interpreter support for GLOBAL_ATOMIC_MIN_I64") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_MIN_U64"),
              "expected interpreter support for GLOBAL_ATOMIC_MIN_U64") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_MAX_I64"),
              "expected interpreter support for GLOBAL_ATOMIC_MAX_I64") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_MAX_U64"),
              "expected interpreter support for GLOBAL_ATOMIC_MAX_U64") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_AND_B64"),
              "expected interpreter support for GLOBAL_ATOMIC_AND_B64") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_OR_B64"),
              "expected interpreter support for GLOBAL_ATOMIC_OR_B64") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_XOR_B64"),
              "expected interpreter support for GLOBAL_ATOMIC_XOR_B64") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_INC_U64"),
              "expected interpreter support for GLOBAL_ATOMIC_INC_U64") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_DEC_U64"),
              "expected interpreter support for GLOBAL_ATOMIC_DEC_U64") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_ADD_F32"),
              "expected interpreter support for GLOBAL_ATOMIC_ADD_F32") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_PK_ADD_F16"),
              "expected interpreter support for GLOBAL_ATOMIC_PK_ADD_F16") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_PK_ADD_BF16"),
              "expected interpreter support for GLOBAL_ATOMIC_PK_ADD_BF16") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_MIN_NUM_F32"),
              "expected interpreter support for GLOBAL_ATOMIC_MIN_NUM_F32") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_MAX_NUM_F32"),
              "expected interpreter support for GLOBAL_ATOMIC_MAX_NUM_F32") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_ORDERED_ADD_B64"),
              "expected interpreter support for GLOBAL_ATOMIC_ORDERED_ADD_B64") ||
      !Expect(interpreter.Supports("DS_NOP"),
              "expected interpreter support for DS_NOP") ||
      !Expect(interpreter.Supports("DS_ADD_RTN_F32"),
              "expected interpreter support for DS_ADD_RTN_F32") ||
      !Expect(interpreter.Supports("DS_ADD_F32"),
              "expected interpreter support for DS_ADD_F32") ||
      !Expect(interpreter.Supports("DS_ADD_RTN_U32"),
              "expected interpreter support for DS_ADD_RTN_U32") ||
      !Expect(interpreter.Supports("DS_ADD_U32"),
              "expected interpreter support for DS_ADD_U32") ||
      !Expect(interpreter.Supports("DS_SUB_RTN_U32"),
              "expected interpreter support for DS_SUB_RTN_U32") ||
      !Expect(interpreter.Supports("DS_SUB_U32"),
              "expected interpreter support for DS_SUB_U32") ||
      !Expect(interpreter.Supports("DS_RSUB_RTN_U32"),
              "expected interpreter support for DS_RSUB_RTN_U32") ||
      !Expect(interpreter.Supports("DS_RSUB_U32"),
              "expected interpreter support for DS_RSUB_U32") ||
      !Expect(interpreter.Supports("DS_INC_RTN_U32"),
              "expected interpreter support for DS_INC_RTN_U32") ||
      !Expect(interpreter.Supports("DS_INC_U32"),
              "expected interpreter support for DS_INC_U32") ||
      !Expect(interpreter.Supports("DS_DEC_RTN_U32"),
              "expected interpreter support for DS_DEC_RTN_U32") ||
      !Expect(interpreter.Supports("DS_DEC_U32"),
              "expected interpreter support for DS_DEC_U32") ||
      !Expect(interpreter.Supports("DS_MIN_RTN_I32"),
              "expected interpreter support for DS_MIN_RTN_I32") ||
      !Expect(interpreter.Supports("DS_MIN_I32"),
              "expected interpreter support for DS_MIN_I32") ||
      !Expect(interpreter.Supports("DS_MIN_RTN_U32"),
              "expected interpreter support for DS_MIN_RTN_U32") ||
      !Expect(interpreter.Supports("DS_MIN_U32"),
              "expected interpreter support for DS_MIN_U32") ||
      !Expect(interpreter.Supports("DS_MAX_RTN_I32"),
              "expected interpreter support for DS_MAX_RTN_I32") ||
      !Expect(interpreter.Supports("DS_MAX_I32"),
              "expected interpreter support for DS_MAX_I32") ||
      !Expect(interpreter.Supports("DS_MAX_RTN_U32"),
              "expected interpreter support for DS_MAX_RTN_U32") ||
      !Expect(interpreter.Supports("DS_MAX_U32"),
              "expected interpreter support for DS_MAX_U32") ||
      !Expect(interpreter.Supports("DS_AND_RTN_B32"),
              "expected interpreter support for DS_AND_RTN_B32") ||
      !Expect(interpreter.Supports("DS_AND_B32"),
              "expected interpreter support for DS_AND_B32") ||
      !Expect(interpreter.Supports("DS_OR_RTN_B32"),
              "expected interpreter support for DS_OR_RTN_B32") ||
      !Expect(interpreter.Supports("DS_OR_B32"),
              "expected interpreter support for DS_OR_B32") ||
      !Expect(interpreter.Supports("DS_XOR_RTN_B32"),
              "expected interpreter support for DS_XOR_RTN_B32") ||
      !Expect(interpreter.Supports("DS_XOR_B32"),
              "expected interpreter support for DS_XOR_B32") ||
      !Expect(interpreter.Supports("DS_LOAD_B32"),
              "expected interpreter support for DS_LOAD_B32") ||
      !Expect(interpreter.Supports("DS_LOAD_B64"),
              "expected interpreter support for DS_LOAD_B64") ||
      !Expect(interpreter.Supports("DS_LOAD_B96"),
              "expected interpreter support for DS_LOAD_B96") ||
      !Expect(interpreter.Supports("DS_LOAD_B128"),
              "expected interpreter support for DS_LOAD_B128") ||
      !Expect(interpreter.Supports("DS_LOAD_I8"),
              "expected interpreter support for DS_LOAD_I8") ||
      !Expect(interpreter.Supports("DS_LOAD_U8"),
              "expected interpreter support for DS_LOAD_U8") ||
      !Expect(interpreter.Supports("DS_LOAD_I16"),
              "expected interpreter support for DS_LOAD_I16") ||
      !Expect(interpreter.Supports("DS_LOAD_U16"),
              "expected interpreter support for DS_LOAD_U16") ||
      !Expect(interpreter.Supports("DS_STORE_B8"),
              "expected interpreter support for DS_STORE_B8") ||
      !Expect(interpreter.Supports("DS_STORE_B16"),
              "expected interpreter support for DS_STORE_B16") ||
      !Expect(interpreter.Supports("DS_STORE_B32"),
              "expected interpreter support for DS_STORE_B32") ||
      !Expect(interpreter.Supports("DS_STORE_B64"),
              "expected interpreter support for DS_STORE_B64") ||
      !Expect(interpreter.Supports("DS_STORE_B96"),
              "expected interpreter support for DS_STORE_B96") ||
      !Expect(interpreter.Supports("DS_STORE_B128"),
              "expected interpreter support for DS_STORE_B128") ||
      !Expect(interpreter.Supports("S_LOAD_B32"),
              "expected interpreter support for S_LOAD_B32") ||
      !Expect(interpreter.Supports("S_LOAD_B64"),
              "expected interpreter support for S_LOAD_B64") ||
      !Expect(interpreter.Supports("S_LOAD_B96"),
              "expected interpreter support for S_LOAD_B96") ||
      !Expect(interpreter.Supports("S_LOAD_B128"),
              "expected interpreter support for S_LOAD_B128") ||
      !Expect(interpreter.Supports("S_LOAD_B256"),
              "expected interpreter support for S_LOAD_B256") ||
      !Expect(interpreter.Supports("S_LOAD_B512"),
              "expected interpreter support for S_LOAD_B512") ||
      !Expect(interpreter.Supports("S_BUFFER_LOAD_B32"),
              "expected interpreter support for S_BUFFER_LOAD_B32") ||
      !Expect(interpreter.Supports("S_BUFFER_LOAD_B64"),
              "expected interpreter support for S_BUFFER_LOAD_B64") ||
      !Expect(interpreter.Supports("S_BUFFER_LOAD_B96"),
              "expected interpreter support for S_BUFFER_LOAD_B96") ||
      !Expect(interpreter.Supports("S_BUFFER_LOAD_B128"),
              "expected interpreter support for S_BUFFER_LOAD_B128") ||
      !Expect(interpreter.Supports("S_BUFFER_LOAD_B256"),
              "expected interpreter support for S_BUFFER_LOAD_B256") ||
      !Expect(interpreter.Supports("S_BUFFER_LOAD_B512"),
              "expected interpreter support for S_BUFFER_LOAD_B512") ||
      !Expect(interpreter.Supports("S_LOAD_I8"),
              "expected interpreter support for S_LOAD_I8") ||
      !Expect(interpreter.Supports("S_LOAD_U8"),
              "expected interpreter support for S_LOAD_U8") ||
      !Expect(interpreter.Supports("S_LOAD_I16"),
              "expected interpreter support for S_LOAD_I16") ||
      !Expect(interpreter.Supports("S_LOAD_U16"),
              "expected interpreter support for S_LOAD_U16") ||
      !Expect(interpreter.Supports("S_BUFFER_LOAD_I8"),
              "expected interpreter support for S_BUFFER_LOAD_I8") ||
      !Expect(interpreter.Supports("S_BUFFER_LOAD_U8"),
              "expected interpreter support for S_BUFFER_LOAD_U8") ||
      !Expect(interpreter.Supports("S_BUFFER_LOAD_I16"),
              "expected interpreter support for S_BUFFER_LOAD_I16") ||
      !Expect(interpreter.Supports("S_BUFFER_LOAD_U16"),
              "expected interpreter support for S_BUFFER_LOAD_U16") ||
      !Expect(interpreter.Supports("S_PREFETCH_INST"),
              "expected interpreter support for S_PREFETCH_INST") ||
      !Expect(interpreter.Supports("S_PREFETCH_INST_PC_REL"),
              "expected interpreter support for S_PREFETCH_INST_PC_REL") ||
      !Expect(interpreter.Supports("S_PREFETCH_DATA"),
              "expected interpreter support for S_PREFETCH_DATA") ||
      !Expect(interpreter.Supports("S_BUFFER_PREFETCH_DATA"),
              "expected interpreter support for S_BUFFER_PREFETCH_DATA") ||
      !Expect(interpreter.Supports("S_PREFETCH_DATA_PC_REL"),
              "expected interpreter support for S_PREFETCH_DATA_PC_REL") ||
      !Expect(interpreter.Supports("S_ATC_PROBE"),
              "expected interpreter support for S_ATC_PROBE") ||
      !Expect(interpreter.Supports("S_ATC_PROBE_BUFFER"),
              "expected interpreter support for S_ATC_PROBE_BUFFER") ||
      !Expect(interpreter.Supports("S_ADD_U32"),
              "expected interpreter support for S_ADD_U32") ||
      !Expect(interpreter.Supports("S_ADD_I32"),
              "expected interpreter support for S_ADD_I32") ||
      !Expect(interpreter.Supports("S_SUB_U32"),
              "expected interpreter support for S_SUB_U32") ||
      !Expect(interpreter.Supports("S_CMP_EQ_U32"),
              "expected interpreter support for S_CMP_EQ_U32") ||
      !Expect(interpreter.Supports("S_CMP_LG_U32"),
              "expected interpreter support for S_CMP_LG_U32") ||
      !Expect(interpreter.Supports("S_CMP_EQ_I32"),
              "expected interpreter support for S_CMP_EQ_I32") ||
      !Expect(interpreter.Supports("S_CMP_GT_I32"),
              "expected interpreter support for S_CMP_GT_I32") ||
      !Expect(interpreter.Supports("S_CMP_LE_U32"),
              "expected interpreter support for S_CMP_LE_U32") ||
      !Expect(interpreter.Supports("S_CMP_GE_I32"),
              "expected interpreter support for S_CMP_GE_I32") ||
      !Expect(interpreter.Supports("S_CMP_LT_I32"),
              "expected interpreter support for S_CMP_LT_I32") ||
      !Expect(interpreter.Supports("S_CMP_GE_U32"),
              "expected interpreter support for S_CMP_GE_U32") ||
      !Expect(interpreter.Supports("S_CMP_LT_U32"),
              "expected interpreter support for S_CMP_LT_U32") ||
      !Expect(interpreter.Supports("S_BRANCH"),
              "expected interpreter support for S_BRANCH") ||
      !Expect(interpreter.Supports("S_CBRANCH_SCC0"),
              "expected interpreter support for S_CBRANCH_SCC0") ||
      !Expect(interpreter.Supports("S_CBRANCH_SCC1"),
              "expected interpreter support for S_CBRANCH_SCC1") ||
      !Expect(interpreter.Supports("S_CBRANCH_VCCZ"),
              "expected interpreter support for S_CBRANCH_VCCZ") ||
      !Expect(interpreter.Supports("S_CBRANCH_VCCNZ"),
              "expected interpreter support for S_CBRANCH_VCCNZ") ||
      !Expect(interpreter.Supports("S_CBRANCH_EXECZ"),
              "expected interpreter support for S_CBRANCH_EXECZ") ||
      !Expect(interpreter.Supports("S_CBRANCH_EXECNZ"),
              "expected interpreter support for S_CBRANCH_EXECNZ") ||
      !Expect(interpreter.Supports("S_MOV_B32"),
              "expected interpreter support for S_MOV_B32") ||
      !Expect(interpreter.Supports("V_NOT_B32"),
              "expected interpreter support for V_NOT_B32") ||
      !Expect(interpreter.Supports("V_CLZ_I32_U32"),
              "expected interpreter support for V_CLZ_I32_U32") ||
      !Expect(interpreter.Supports("V_CLS_I32"),
              "expected interpreter support for V_CLS_I32") ||
      !Expect(interpreter.Supports("V_CMP_EQ_U32"),
              "expected interpreter support for V_CMP_EQ_U32") ||
      !Expect(interpreter.Supports("V_CMPX_EQ_U32"),
              "expected interpreter support for V_CMPX_EQ_U32") ||
      !Expect(interpreter.Supports("V_CMP_EQ_I16"),
              "expected interpreter support for V_CMP_EQ_I16") ||
      !Expect(interpreter.Supports("V_CMPX_EQ_U16"),
              "expected interpreter support for V_CMPX_EQ_U16") ||
      !Expect(interpreter.Supports("V_CMP_EQ_F16"),
              "expected interpreter support for V_CMP_EQ_F16") ||
      !Expect(interpreter.Supports("V_CMPX_CLASS_F16"),
              "expected interpreter support for V_CMPX_CLASS_F16") ||
      !Expect(interpreter.Supports("V_CMP_EQ_F32"),
              "expected interpreter support for V_CMP_EQ_F32") ||
      !Expect(interpreter.Supports("V_CMP_O_F32"),
              "expected interpreter support for V_CMP_O_F32") ||
      !Expect(interpreter.Supports("V_CMPX_CLASS_F32"),
              "expected interpreter support for V_CMPX_CLASS_F32") ||
      !Expect(interpreter.Supports("V_CMPX_U_F32"),
              "expected interpreter support for V_CMPX_U_F32") ||
      !Expect(interpreter.Supports("V_CMP_EQ_F64"),
              "expected interpreter support for V_CMP_EQ_F64") ||
      !Expect(interpreter.Supports("V_CMPX_CLASS_F64"),
              "expected interpreter support for V_CMPX_CLASS_F64") ||
      !Expect(interpreter.Supports("V_CMP_EQ_I64"),
              "expected interpreter support for V_CMP_EQ_I64") ||
      !Expect(interpreter.Supports("V_CMPX_EQ_U64"),
              "expected interpreter support for V_CMPX_EQ_U64") ||
      !Expect(interpreter.Supports("V_CMP_GE_I32"),
              "expected interpreter support for V_CMP_GE_I32") ||
      !Expect(interpreter.Supports("V_CMPX_GE_I32"),
              "expected interpreter support for V_CMPX_GE_I32") ||
      !Expect(interpreter.Supports("V_NOP"),
              "expected interpreter support for V_NOP") ||
      !Expect(interpreter.Supports("V_PIPEFLUSH"),
              "expected interpreter support for V_PIPEFLUSH") ||
      !Expect(interpreter.Supports("V_MOV_B16"),
              "expected interpreter support for V_MOV_B16") ||
      !Expect(interpreter.Supports("V_PERMLANE64_B32"),
              "expected interpreter support for V_PERMLANE64_B32") ||
      !Expect(interpreter.Supports("V_SWAP_B32"),
              "expected interpreter support for V_SWAP_B32") ||
      !Expect(interpreter.Supports("V_SWAP_B16"),
              "expected interpreter support for V_SWAP_B16") ||
      !Expect(interpreter.Supports("V_NOT_B16"),
              "expected interpreter support for V_NOT_B16") ||
      !Expect(interpreter.Supports("V_BFREV_B32"),
              "expected interpreter support for V_BFREV_B32") ||
      !Expect(interpreter.Supports("V_CVT_F32_UBYTE0"),
              "expected interpreter support for V_CVT_F32_UBYTE0") ||
      !Expect(interpreter.Supports("V_CVT_F32_UBYTE3"),
              "expected interpreter support for V_CVT_F32_UBYTE3") ||
      !Expect(interpreter.Supports("V_CVT_F32_I32"),
              "expected interpreter support for V_CVT_F32_I32") ||
      !Expect(interpreter.Supports("V_CVT_F32_FP8"),
              "expected interpreter support for V_CVT_F32_FP8") ||
      !Expect(interpreter.Supports("V_CVT_F32_BF8"),
              "expected interpreter support for V_CVT_F32_BF8") ||
      !Expect(interpreter.Supports("V_CVT_PK_F32_FP8"),
              "expected interpreter support for V_CVT_PK_F32_FP8") ||
      !Expect(interpreter.Supports("V_CVT_PK_F32_BF8"),
              "expected interpreter support for V_CVT_PK_F32_BF8") ||
      !Expect(interpreter.Supports("V_CVT_NEAREST_I32_F32"),
              "expected interpreter support for V_CVT_NEAREST_I32_F32") ||
      !Expect(interpreter.Supports("V_CVT_F16_F32"),
              "expected interpreter support for V_CVT_F16_F32") ||
      !Expect(interpreter.Supports("V_CVT_F16_I16"),
              "expected interpreter support for V_CVT_F16_I16") ||
      !Expect(interpreter.Supports("V_CVT_F16_U16"),
              "expected interpreter support for V_CVT_F16_U16") ||
      !Expect(interpreter.Supports("V_CVT_F32_F16"),
              "expected interpreter support for V_CVT_F32_F16") ||
      !Expect(interpreter.Supports("V_CVT_I16_F16"),
              "expected interpreter support for V_CVT_I16_F16") ||
      !Expect(interpreter.Supports("V_CVT_U16_F16"),
              "expected interpreter support for V_CVT_U16_F16") ||
      !Expect(interpreter.Supports("V_SAT_PK_U8_I16"),
              "expected interpreter support for V_SAT_PK_U8_I16") ||
      !Expect(interpreter.Supports("V_CVT_OFF_F32_I4"),
              "expected interpreter support for V_CVT_OFF_F32_I4") ||
      !Expect(interpreter.Supports("V_CVT_NORM_I16_F16"),
              "expected interpreter support for V_CVT_NORM_I16_F16") ||
      !Expect(interpreter.Supports("V_CVT_NORM_U16_F16"),
              "expected interpreter support for V_CVT_NORM_U16_F16") ||
      !Expect(interpreter.Supports("V_RCP_F16"),
              "expected interpreter support for V_RCP_F16") ||
      !Expect(interpreter.Supports("V_RSQ_F16"),
              "expected interpreter support for V_RSQ_F16") ||
      !Expect(interpreter.Supports("V_SQRT_F16"),
              "expected interpreter support for V_SQRT_F16") ||
      !Expect(interpreter.Supports("V_EXP_F16"),
              "expected interpreter support for V_EXP_F16") ||
      !Expect(interpreter.Supports("V_LOG_F16"),
              "expected interpreter support for V_LOG_F16") ||
      !Expect(interpreter.Supports("V_SIN_F16"),
              "expected interpreter support for V_SIN_F16") ||
      !Expect(interpreter.Supports("V_COS_F16"),
              "expected interpreter support for V_COS_F16") ||
      !Expect(interpreter.Supports("V_FREXP_EXP_I16_F16"),
              "expected interpreter support for V_FREXP_EXP_I16_F16") ||
      !Expect(interpreter.Supports("V_FREXP_MANT_F16"),
              "expected interpreter support for V_FREXP_MANT_F16") ||
      !Expect(interpreter.Supports("V_FRACT_F16"),
              "expected interpreter support for V_FRACT_F16") ||
      !Expect(interpreter.Supports("V_TRUNC_F16"),
              "expected interpreter support for V_TRUNC_F16") ||
      !Expect(interpreter.Supports("V_CEIL_F16"),
              "expected interpreter support for V_CEIL_F16") ||
      !Expect(interpreter.Supports("V_RNDNE_F16"),
              "expected interpreter support for V_RNDNE_F16") ||
      !Expect(interpreter.Supports("V_FLOOR_F16"),
              "expected interpreter support for V_FLOOR_F16") ||
      !Expect(interpreter.Supports("V_ADD_F16"),
              "expected interpreter support for V_ADD_F16") ||
      !Expect(interpreter.Supports("V_SUB_F16"),
              "expected interpreter support for V_SUB_F16") ||
      !Expect(interpreter.Supports("V_SUBREV_F16"),
              "expected interpreter support for V_SUBREV_F16") ||
      !Expect(interpreter.Supports("V_MUL_F16"),
              "expected interpreter support for V_MUL_F16") ||
      !Expect(interpreter.Supports("V_CVT_PK_RTZ_F16_F32"),
              "expected interpreter support for V_CVT_PK_RTZ_F16_F32") ||
      !Expect(interpreter.Supports("V_LDEXP_F16"),
              "expected interpreter support for V_LDEXP_F16") ||
      !Expect(interpreter.Supports("V_MIN_NUM_F16"),
              "expected interpreter support for V_MIN_NUM_F16") ||
      !Expect(interpreter.Supports("V_MAX_NUM_F16"),
              "expected interpreter support for V_MAX_NUM_F16") ||
      !Expect(interpreter.Supports("V_ADD_F32"),
              "expected interpreter support for V_ADD_F32") ||
      !Expect(interpreter.Supports("V_SUB_F32"),
              "expected interpreter support for V_SUB_F32") ||
      !Expect(interpreter.Supports("V_SUBREV_F32"),
              "expected interpreter support for V_SUBREV_F32") ||
      !Expect(interpreter.Supports("V_MUL_F32"),
              "expected interpreter support for V_MUL_F32") ||
      !Expect(interpreter.Supports("V_MIN_NUM_F32"),
              "expected interpreter support for V_MIN_NUM_F32") ||
      !Expect(interpreter.Supports("V_MAX_NUM_F32"),
              "expected interpreter support for V_MAX_NUM_F32") ||
      !Expect(interpreter.Supports("V_ADD_F64"),
              "expected interpreter support for V_ADD_F64") ||
      !Expect(interpreter.Supports("V_MUL_F64"),
              "expected interpreter support for V_MUL_F64") ||
      !Expect(interpreter.Supports("V_MIN_NUM_F64"),
              "expected interpreter support for V_MIN_NUM_F64") ||
      !Expect(interpreter.Supports("V_MAX_NUM_F64"),
              "expected interpreter support for V_MAX_NUM_F64") ||
      !Expect(interpreter.Supports("V_XNOR_B32"),
              "expected interpreter support for V_XNOR_B32") ||
      !Expect(interpreter.Supports("V_MUL_I32_I24"),
              "expected interpreter support for V_MUL_I32_I24") ||
      !Expect(interpreter.Supports("V_MUL_HI_I32_I24"),
              "expected interpreter support for V_MUL_HI_I32_I24") ||
      !Expect(interpreter.Supports("V_MUL_U32_U24"),
              "expected interpreter support for V_MUL_U32_U24") ||
      !Expect(interpreter.Supports("V_MUL_HI_U32_U24"),
              "expected interpreter support for V_MUL_HI_U32_U24") ||
      !Expect(interpreter.Supports("V_LSHLREV_B64"),
              "expected interpreter support for V_LSHLREV_B64") ||
      !Expect(interpreter.Supports("V_CVT_I32_I16"),
              "expected interpreter support for V_CVT_I32_I16") ||
      !Expect(interpreter.Supports("V_CVT_U32_U16"),
              "expected interpreter support for V_CVT_U32_U16") ||
      !Expect(interpreter.Supports("V_CVT_F32_U32"),
              "expected interpreter support for V_CVT_F32_U32") ||
      !Expect(interpreter.Supports("V_CVT_F64_F32"),
              "expected interpreter support for V_CVT_F64_F32") ||
      !Expect(interpreter.Supports("V_CVT_I32_F64"),
              "expected interpreter support for V_CVT_I32_F64") ||
      !Expect(interpreter.Supports("V_RCP_F32"),
              "expected interpreter support for V_RCP_F32") ||
      !Expect(interpreter.Supports("V_LOG_F32"),
              "expected interpreter support for V_LOG_F32") ||
      !Expect(interpreter.Supports("V_SQRT_F64"),
              "expected interpreter support for V_SQRT_F64") ||
      !Expect(interpreter.Supports("V_FREXP_EXP_I32_F32"),
              "expected interpreter support for V_FREXP_EXP_I32_F32") ||
      !Expect(interpreter.Supports("V_FRACT_F64"),
              "expected interpreter support for V_FRACT_F64") ||
      !Expect(interpreter.Supports("V_TRUNC_F32"),
              "expected interpreter support for V_TRUNC_F32") ||
      !Expect(interpreter.Supports("V_FLOOR_F64"),
              "expected interpreter support for V_FLOOR_F64") ||
      !Expect(interpreter.Supports("V_CVT_U32_F32"),
              "expected interpreter support for V_CVT_U32_F32") ||
      !Expect(interpreter.Supports("V_CVT_I32_F32"),
              "expected interpreter support for V_CVT_I32_F32") ||
      !Expect(interpreter.Supports("V_READFIRSTLANE_B32"),
              "expected interpreter support for V_READFIRSTLANE_B32") ||
      !Expect(interpreter.Supports("V_MOVRELD_B32"),
              "expected interpreter support for V_MOVRELD_B32") ||
      !Expect(interpreter.Supports("V_MOVRELS_B32"),
              "expected interpreter support for V_MOVRELS_B32") ||
      !Expect(interpreter.Supports("V_MOVRELSD_B32"),
              "expected interpreter support for V_MOVRELSD_B32") ||
      !Expect(interpreter.Supports("V_MOVRELSD_2_B32"),
              "expected interpreter support for V_MOVRELSD_2_B32") ||
      !Expect(interpreter.Supports("V_SWAPREL_B32"),
              "expected interpreter support for V_SWAPREL_B32") ||
      !Expect(interpreter.Supports("V_ADD_U32"),
              "expected interpreter support for V_ADD_U32") ||
      !Expect(interpreter.Supports("V_SUB_U32"),
              "expected interpreter support for V_SUB_U32") ||
      !Expect(interpreter.Supports("V_SUBREV_U32"),
              "expected interpreter support for V_SUBREV_U32") ||
      !Expect(interpreter.Supports("V_CNDMASK_B32"),
              "expected interpreter support for V_CNDMASK_B32") ||
      !Expect(interpreter.Supports("V_MIN_I32"),
              "expected interpreter support for V_MIN_I32") ||
      !Expect(interpreter.Supports("V_LSHRREV_B32"),
              "expected interpreter support for V_LSHRREV_B32") ||
      !Expect(interpreter.Supports("V_XOR_B32"),
              "expected interpreter support for V_XOR_B32") ||
      !Expect(interpreter.Supports("V_MOV_B32"),
              "expected interpreter support for V_MOV_B32") ||
      !Expect(interpreter.Supports("V_MUL_DX9_ZERO_F32"),
              "expected interpreter support for V_MUL_DX9_ZERO_F32") ||
      !Expect(interpreter.Supports("V_FMAC_F32"),
              "expected interpreter support for V_FMAC_F32") ||
      !Expect(interpreter.Supports("V_FMAC_F16"),
              "expected interpreter support for V_FMAC_F16") ||
      !Expect(interpreter.Supports("V_FMAMK_F16"),
              "expected interpreter support for V_FMAMK_F16") ||
      !Expect(interpreter.Supports("V_FMAAK_F16"),
              "expected interpreter support for V_FMAAK_F16") ||
      !Expect(interpreter.Supports("V_PK_FMAC_F16"),
              "expected interpreter support for V_PK_FMAC_F16") ||
      !Expect(interpreter.Supports("V_ADD_CO_CI_U32"),
              "expected interpreter support for V_ADD_CO_CI_U32") ||
      !Expect(interpreter.Supports("V_SUB_CO_CI_U32"),
              "expected interpreter support for V_SUB_CO_CI_U32") ||
      !Expect(interpreter.Supports("V_SUBREV_CO_CI_U32"),
              "expected interpreter support for V_SUBREV_CO_CI_U32") ||
      !Expect(!interpreter.Supports("V_DOT2_F32_F16"),
              "expected interpreter to reject unsupported seed opcode") ||
      !Expect(interpreter.CarryOverFamilyFocus().size() == 7u,
              "expected carry-over family focus list") ||
      !Expect(interpreter.Rdna4DeltaFamilyFocus().size() == 10u,
              "expected RDNA4 delta family focus list")) {
    return 1;
  }

  const std::array<DecodedInstruction, 3> supported_program{
      DecodedInstruction::Unary("S_MOVK_I32", InstructionOperand::Sgpr(1),
                                InstructionOperand::Imm32(7u)),
      DecodedInstruction::Binary("S_ADD_U32", InstructionOperand::Sgpr(2),
                                 InstructionOperand::Sgpr(1),
                                 InstructionOperand::Imm32(5u)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  std::vector<Gfx1201CompiledInstruction> compiled_program;
  if (!Expect(interpreter.CompileProgram(supported_program, &compiled_program,
                                         &error_message),
              "expected compile success for executable seed slice") ||
      !Expect(compiled_program.size() == supported_program.size(),
              "expected compiled instruction count")) {
    return 1;
  }

  WaveExecutionState state;
  if (!Expect(interpreter.ExecuteProgram(supported_program, &state, &error_message),
              "expected decoded execution success for executable seed slice") ||
      !Expect(state.lane_count == 32u,
              "expected gfx1201 execution to normalize to wave32") ||
      !Expect(state.exec_mask == 0xffffffffULL,
              "expected gfx1201 execution to clamp exec to wave32") ||
      !Expect(state.sgprs[1] == 7u, "expected decoded execution to write SGPR") ||
      !Expect(state.sgprs[2] == 12u, "expected decoded execution to add into SGPR") ||
      !Expect(state.halted, "expected decoded execution to halt")) {
    return 1;
  }

  const std::array<DecodedInstruction, 1> unsupported_program{
      DecodedInstruction::Binary("V_DOT2_F32_F16", InstructionOperand::Vgpr(0),
                                 InstructionOperand::Vgpr(1),
                                 InstructionOperand::Vgpr(2)),
  };
  if (!Expect(!interpreter.CompileProgram(unsupported_program, &compiled_program,
                                          &error_message),
              "expected unsupported program compile failure") ||
      !Expect(error_message.find("V_DOT2_F32_F16 is not in the executable seed slice") !=
                  std::string::npos,
              "expected unsupported opcode message") ||
      !Expect(error_message.find("RDNA4 delta families") != std::string::npos,
              "expected remaining bring-up summary in error")) {
    return 1;
  }

  return 0;
}
