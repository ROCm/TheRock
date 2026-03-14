#include <array>
#include <cstdint>
#include <iostream>
#include <string>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/gfx1201/binary_decoder.h"
#include "lib/sim/isa/gfx1201/opcode_selector.h"

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

constexpr std::uint32_t MakeSopc(std::uint32_t op,
                                 std::uint32_t ssrc0,
                                 std::uint32_t ssrc1) {
  std::uint32_t word = 0;
  word = SetBits(word, 0x17e, 23, 9);
  word = SetBits(word, op, 16, 7);
  word = SetBits(word, ssrc1, 8, 8);
  word = SetBits(word, ssrc0, 0, 8);
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

constexpr std::uint32_t MakeSop1(std::uint32_t op,
                                 std::uint32_t sdst,
                                 std::uint32_t ssrc0) {
  std::uint32_t word = 0;
  word = SetBits(word, 0x17d, 23, 9);
  word = SetBits(word, sdst, 16, 7);
  word = SetBits(word, op, 8, 8);
  word = SetBits(word, ssrc0, 0, 8);
  return word;
}

constexpr std::uint32_t MakeSop2(std::uint32_t op,
                                 std::uint32_t sdst,
                                 std::uint32_t ssrc0,
                                 std::uint32_t ssrc1) {
  std::uint32_t word = 0;
  word = SetBits(word, 0x2, 30, 2);
  word = SetBits(word, op, 23, 7);
  word = SetBits(word, sdst, 16, 7);
  word = SetBits(word, ssrc1, 8, 8);
  word = SetBits(word, ssrc0, 0, 8);
  return word;
}

constexpr std::uint32_t MakeVop1(std::uint32_t op,
                                 std::uint32_t vdst,
                                 std::uint32_t src0) {
  std::uint32_t word = 0;
  word = SetBits(word, 0x3f, 25, 7);
  word = SetBits(word, vdst, 17, 8);
  word = SetBits(word, op, 9, 8);
  word = SetBits(word, src0, 0, 9);
  return word;
}

constexpr std::uint32_t MakeVop2(std::uint32_t op,
                                 std::uint32_t vdst,
                                 std::uint32_t src0,
                                 std::uint32_t vsrc1) {
  std::uint32_t word = 0;
  word = SetBits(word, 0x0, 31, 1);
  word = SetBits(word, op, 25, 6);
  word = SetBits(word, vdst, 17, 8);
  word = SetBits(word, vsrc1, 9, 8);
  word = SetBits(word, src0, 0, 9);
  return word;
}

constexpr std::uint32_t MakeVopc(std::uint32_t op,
                                 std::uint32_t src0,
                                 std::uint32_t vsrc1) {
  std::uint32_t word = 0;
  word = SetBits(word, 0x3e, 25, 7);
  word = SetBits(word, op, 17, 8);
  word = SetBits(word, vsrc1, 9, 8);
  word = SetBits(word, src0, 0, 9);
  return word;
}

std::array<std::uint32_t, 2> MakeVop3(std::uint32_t op,
                                      std::uint32_t vdst,
                                      std::uint32_t src0,
                                      std::uint32_t src1,
                                      std::uint32_t src2 = 0) {
  std::uint64_t word = 0;
  word |= static_cast<std::uint64_t>(vdst & 0xffu) << 0;
  word |= static_cast<std::uint64_t>(op & 0x3ffu) << 16;
  word |= static_cast<std::uint64_t>(0x34u) << 26;
  word |= static_cast<std::uint64_t>(src0 & 0x1ffu) << 32;
  word |= static_cast<std::uint64_t>(src1 & 0x1ffu) << 41;
  word |= static_cast<std::uint64_t>(src2 & 0x1ffu) << 50;
  return {static_cast<std::uint32_t>(word),
          static_cast<std::uint32_t>(word >> 32)};
}

std::array<std::uint32_t, 2> MakeSmem(std::uint32_t op,
                                      std::uint32_t sdata,
                                      std::uint32_t sbase_start,
                                      bool imm,
                                      std::uint32_t offset_or_soffset,
                                      bool soffset_en = false) {
  std::uint64_t word = 0;
  word |= static_cast<std::uint64_t>(0x30u) << 26;
  word |= static_cast<std::uint64_t>(sbase_start >> 1) << 0;
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

std::array<std::uint32_t, 2> MakeFlatLike(std::uint32_t op) {
  std::uint64_t word = 0;
  word |= static_cast<std::uint64_t>(55u) << 26;
  word |= static_cast<std::uint64_t>(op & 0x7fu) << 18;
  return {static_cast<std::uint32_t>(word),
          static_cast<std::uint32_t>(word >> 32)};
}

}  // namespace

int main() {
  using namespace mirage::sim::isa;

  const auto rules = GetGfx1201Phase0ComputeOpcodeSelectorRules();
  if (!Expect(rules.size() == 12u, "expected 12 selector rules") ||
      !Expect(rules.front().encoding_name == "ENC_VGLOBAL",
              "expected VGLOBAL to route before other flat-like forms") ||
      !Expect(FindGfx1201Phase0ComputeOpcodeSelectorRule("ENC_VOP3") != nullptr,
              "expected ENC_VOP3 selector rule") ||
      !Expect(FindGfx1201Phase0ComputeOpcodeSelectorRule("ENC_VIMAGE") ==
                  nullptr,
              "expected no selector rule for graphics encoding")) {
    return 1;
  }

  struct RouteCase {
    const char* encoding_name;
    const char* instruction_name;
    std::uint32_t opcode;
    std::span<const std::uint32_t> words;
    std::size_t words_required;
  };

  const std::uint32_t sopp_word = MakeSopp(48u);
  const std::uint32_t sop1_word = MakeSop1(0u, 1u, 2u);
  const std::uint32_t sop2_word = MakeSop2(22u, 1u, 2u, 3u);
  const std::uint32_t sopc_word = MakeSopc(6u, 1u, 2u);
  const std::uint32_t sopk_word = MakeSopk(0u, 1u, 7u);
  const auto smem_words = MakeSmem(0u, 1u, 0u, true, 0u);
  const std::uint32_t vop1_word = MakeVop1(1u, 1u, 2u);
  const std::uint32_t vop2_word = MakeVop2(3u, 1u, 2u, 3u);
  const std::array<std::uint32_t, 2> vop2_literal_words{
      MakeVop2(55u, 1u, 2u, 3u), 0x00003c00u};
  const std::uint32_t vopc_word = MakeVopc(18u, 1u, 2u);
  const auto vop3_words = MakeVop3(597u, 1u, 2u, 3u, 4u);
  const auto ds_words = MakeDs(0u, 1u, 2u, 3u, 4u, 0u);
  const auto global_words = MakeGlobal(20u, 1u, 2u, 3u, 4u, 0u);

  const std::array<RouteCase, 12> route_cases{{
      {"ENC_SOPP", "S_ENDPGM", 48u, std::span<const std::uint32_t>(&sopp_word, 1),
       1u},
      {"ENC_SOP1", "S_MOV_B32", 0u, std::span<const std::uint32_t>(&sop1_word, 1),
       1u},
      {"ENC_SOP2", "S_AND_B32", 22u, std::span<const std::uint32_t>(&sop2_word, 1),
       1u},
      {"ENC_SOPC", "S_CMP_EQ_U32", 6u,
       std::span<const std::uint32_t>(&sopc_word, 1), 1u},
      {"ENC_SOPK", "S_MOVK_I32", 0u, std::span<const std::uint32_t>(&sopk_word, 1),
       1u},
      {"ENC_SMEM", "S_LOAD_B32", 0u,
       std::span<const std::uint32_t>(smem_words.data(), smem_words.size()), 2u},
      {"ENC_VOP1", "V_MOV_B32", 1u, std::span<const std::uint32_t>(&vop1_word, 1),
       1u},
      {"ENC_VOP2", "V_ADD_F32", 3u, std::span<const std::uint32_t>(&vop2_word, 1),
       1u},
      {"ENC_VOPC", "V_CMP_EQ_F32", 18u,
       std::span<const std::uint32_t>(&vopc_word, 1), 1u},
      {"ENC_VOP3", "V_ADD3_U32", 597u,
       std::span<const std::uint32_t>(vop3_words.data(), vop3_words.size()), 2u},
      {"ENC_VDS", "DS_ADD_U32", 0u,
       std::span<const std::uint32_t>(ds_words.data(), ds_words.size()), 2u},
      {"ENC_VGLOBAL", "GLOBAL_LOAD_B32", 20u,
       std::span<const std::uint32_t>(global_words.data(), global_words.size()), 2u},
  }};

  for (const RouteCase& route_case : route_cases) {
    Gfx1201OpcodeRoute route;
    std::string error_message;
    if (!Expect(SelectGfx1201Phase0ComputeOpcodeRoute(route_case.words, &route,
                                                      &error_message),
                "expected route selection success") ||
        !Expect(route.status == Gfx1201OpcodeRouteStatus::kMatchedSeedEntry,
                "expected seeded opcode route") ||
        !Expect(route.selector_rule != nullptr, "expected selector rule") ||
        !Expect(route.seed_encoding != nullptr, "expected seed encoding") ||
        !Expect(route.seed_entry != nullptr, "expected seed entry") ||
        !Expect(route.selector_rule->encoding_name == route_case.encoding_name,
                "expected selected encoding") ||
        !Expect(route.seed_entry->instruction_name == route_case.instruction_name,
                "expected selected instruction") ||
        !Expect(route.opcode == route_case.opcode, "expected selected opcode") ||
        !Expect(route.words_required == route_case.words_required,
                "expected selected word width")) {
      return 1;
    }
  }

  Gfx1201OpcodeRoute vop3_partial_route;
  std::string error_message;
  if (!Expect(SelectGfx1201Phase0ComputeOpcodeRoute(
                  std::span<const std::uint32_t>(vop3_words.data(), 1),
                  &vop3_partial_route, &error_message),
              "expected one-word VOP3 route to classify") ||
      !Expect(vop3_partial_route.status ==
                  Gfx1201OpcodeRouteStatus::kNeedsMoreWords,
              "expected VOP3 to require two dwords") ||
      !Expect(vop3_partial_route.selector_rule != nullptr &&
                  vop3_partial_route.selector_rule->encoding_name == "ENC_VOP3",
              "expected VOP3 rule on partial route") ||
      !Expect(vop3_partial_route.opcode == 597u,
              "expected VOP3 opcode extraction on partial route")) {
    return 1;
  }

  Gfx1201OpcodeRoute vop2_literal_partial_route;
  if (!Expect(SelectGfx1201Phase0ComputeOpcodeRoute(
                  std::span<const std::uint32_t>(vop2_literal_words.data(), 1),
                  &vop2_literal_partial_route, &error_message),
              "expected one-word VOP2 literal route to classify") ||
      !Expect(vop2_literal_partial_route.status ==
                  Gfx1201OpcodeRouteStatus::kNeedsMoreWords,
              "expected VOP2 literal route to require two dwords") ||
      !Expect(vop2_literal_partial_route.seed_entry != nullptr &&
                  vop2_literal_partial_route.seed_entry->instruction_name ==
                      "V_FMAMK_F16",
              "expected V_FMAMK_F16 seed entry on partial literal route") ||
      !Expect(vop2_literal_partial_route.words_required == 2u,
              "expected VOP2 literal route to advertise two dwords")) {
    return 1;
  }

  Gfx1201OpcodeRoute vop2_literal_route;
  if (!Expect(SelectGfx1201Phase0ComputeOpcodeRoute(
                  std::span<const std::uint32_t>(vop2_literal_words.data(),
                                                 vop2_literal_words.size()),
                  &vop2_literal_route, &error_message),
              "expected VOP2 literal route selection success") ||
      !Expect(vop2_literal_route.status ==
                  Gfx1201OpcodeRouteStatus::kMatchedSeedEntry,
              "expected seeded VOP2 literal route") ||
      !Expect(vop2_literal_route.selector_rule != nullptr &&
                  vop2_literal_route.selector_rule->encoding_name == "ENC_VOP2",
              "expected ENC_VOP2 selector rule for literal route") ||
      !Expect(vop2_literal_route.opcode == 55u,
              "expected VOP2 literal opcode extraction")) {
    return 1;
  }

  const std::uint32_t unknown_sopp_word = MakeSopp(127u);
  Gfx1201OpcodeRoute unknown_sopp_route;
  if (!Expect(SelectGfx1201Phase0ComputeOpcodeRoute(
                  std::span<const std::uint32_t>(&unknown_sopp_word, 1),
                  &unknown_sopp_route, &error_message),
              "expected unknown SOPP opcode to classify by encoding") ||
      !Expect(unknown_sopp_route.status ==
                  Gfx1201OpcodeRouteStatus::kMatchedEncodingOnly,
              "expected unknown SOPP opcode to miss seed entry") ||
      !Expect(!unknown_sopp_route.HasSeedEntry(),
              "expected unknown SOPP opcode to have no seed entry")) {
    return 1;
  }

  const auto flat_words = MakeFlatLike(1u);
  Gfx1201OpcodeRoute unsupported_route;
  if (!Expect(!SelectGfx1201Phase0ComputeOpcodeRoute(
                  std::span<const std::uint32_t>(flat_words.data(),
                                                 flat_words.size()),
                  &unsupported_route, &error_message),
              "expected flat encoding to be out of phase-0 scope") ||
      !Expect(unsupported_route.status ==
                  Gfx1201OpcodeRouteStatus::kUnsupportedEncoding,
              "expected unsupported route status") ||
      !Expect(error_message.find("unsupported or unknown gfx1201 phase-0 compute "
                                 "encoding") != std::string::npos,
              "expected unsupported encoding error")) {
    return 1;
  }

  Gfx1201BinaryDecoder decoder;
  Gfx1201OpcodeRoute decoder_route;
  if (!Expect(decoder.SelectPhase0ComputeRoute(route_cases[9].words,
                                               &decoder_route, &error_message),
              "expected binary decoder route wrapper") ||
      !Expect(decoder_route.seed_entry != nullptr &&
                  decoder_route.seed_entry->instruction_name == "V_ADD3_U32",
              "expected decoder route to expose V_ADD3_U32") ||
      !Expect(decoder.Phase0ComputeSelectorRules().size() == rules.size(),
              "expected binary decoder selector rules")) {
    return 1;
  }

  DecodedInstruction decoded_instruction;
  std::size_t words_consumed = 0;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(&sopp_word, 1),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected S_ENDPGM decode success after route") ||
      !Expect(decoded_instruction.opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM opcode") ||
      !Expect(words_consumed == 1u, "expected one consumed dword")) {
    return 1;
  }

  if (!Expect(!decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(vop3_words.data(), 1),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected partial VOP3 decode failure") ||
      !Expect(error_message.find("ENC_VOP3 opcode 597") != std::string::npos,
              "expected routed VOP3 message") ||
      !Expect(error_message.find("needs 2 dwords") != std::string::npos,
              "expected partial VOP3 word-count error")) {
    return 1;
  }

  if (!Expect(!decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(vop2_literal_words.data(), 1),
                  &decoded_instruction, &words_consumed, &error_message),
              "expected partial VOP2 literal decode failure") ||
      !Expect(error_message.find("ENC_VOP2 opcode 55") != std::string::npos,
              "expected routed VOP2 literal message") ||
      !Expect(error_message.find("V_FMAMK_F16") != std::string::npos,
              "expected V_FMAMK_F16 route detail") ||
      !Expect(error_message.find("needs 2 dwords") != std::string::npos,
              "expected partial VOP2 literal word-count error")) {
    return 1;
  }

  return 0;
}
