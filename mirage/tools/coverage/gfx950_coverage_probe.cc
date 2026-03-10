#include <array>
#include <cstdint>
#include <iostream>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "lib/sim/isa/decoded_instruction.h"
#include "lib/sim/isa/gfx950_binary_decoder.h"
#include "lib/sim/isa/gfx950_interpreter.h"
#include "lib/sim/isa/instruction_catalog.h"

namespace {

using mirage::sim::isa::DecodedInstruction;
using mirage::sim::isa::Gfx950BinaryDecoder;
using mirage::sim::isa::Gfx950Interpreter;
using mirage::sim::isa::InstructionEncodingSpec;
using mirage::sim::isa::InstructionSpec;

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

std::array<std::uint32_t, 2> MakeVop3Sdst(std::uint32_t op,
                                          std::uint32_t vdst,
                                          std::uint32_t sdst,
                                          std::uint32_t src0,
                                          std::uint32_t src1,
                                          std::uint32_t src2 = 0) {
  std::uint64_t word = 0;
  word |= static_cast<std::uint64_t>(vdst & 0xffu) << 0;
  word |= static_cast<std::uint64_t>(sdst & 0x7fu) << 8;
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

std::array<std::uint32_t, 2> MakeFlat(std::uint32_t op,
                                      std::uint32_t vdst,
                                      std::uint32_t addr,
                                      std::uint32_t data,
                                      std::uint32_t offset) {
  std::uint64_t word = 0;
  word |= static_cast<std::uint64_t>(55u) << 26;
  word |= static_cast<std::uint64_t>(offset & 0xfffu) << 0;
  word |= static_cast<std::uint64_t>(op & 0x7fu) << 18;
  word |= static_cast<std::uint64_t>(addr & 0xffu) << 32;
  word |= static_cast<std::uint64_t>(data & 0xffu) << 40;
  word |= static_cast<std::uint64_t>(vdst & 0xffu) << 56;
  return {static_cast<std::uint32_t>(word),
          static_cast<std::uint32_t>(word >> 32)};
}

std::array<std::uint32_t, 2> MakeDs(std::uint32_t op,
                                    std::uint32_t vdst,
                                    std::uint32_t addr,
                                    std::uint32_t data0,
                                    std::uint32_t data1,
                                    std::uint32_t offset0) {
  std::uint64_t word = 0;
  word |= static_cast<std::uint64_t>(offset0 & 0xffu) << 0;
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

std::array<std::uint32_t, 2> MakeGlobalAtomic(std::uint32_t op,
                                              bool return_prior_value,
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
  word |= static_cast<std::uint64_t>(return_prior_value ? 1u : 0u) << 16;
  word |= static_cast<std::uint64_t>(op & 0x7fu) << 18;
  word |= static_cast<std::uint64_t>(addr & 0xffu) << 32;
  word |= static_cast<std::uint64_t>(data & 0xffu) << 40;
  word |= static_cast<std::uint64_t>(saddr & 0x7fu) << 48;
  word |= static_cast<std::uint64_t>(vdst & 0xffu) << 56;
  return {static_cast<std::uint32_t>(word),
          static_cast<std::uint32_t>(word >> 32)};
}

bool IsVectorCarryOutBinaryOpcode(std::string_view opcode) {
  return opcode == "V_ADD_CO_U32" || opcode == "V_SUB_CO_U32" ||
         opcode == "V_SUBREV_CO_U32";
}

bool IsVectorCarryInBinaryOpcode(std::string_view opcode) {
  return opcode == "V_ADDC_CO_U32" || opcode == "V_SUBB_CO_U32" ||
         opcode == "V_SUBBREV_CO_U32";
}

std::vector<std::string_view> ExpectedDecodedOpcodes(
    std::string_view instruction_name) {
  if (instruction_name == "S_MOVK_I32") {
    return {"S_MOV_B32"};
  }
  if (instruction_name == "S_CMOVK_I32") {
    return {"S_CMOV_B32"};
  }
  if (instruction_name == "S_ADDK_I32") {
    return {"S_ADD_U32"};
  }
  if (instruction_name == "S_MULK_I32") {
    return {"S_MUL_I32"};
  }
  if (instruction_name == "S_CMPK_EQ_I32") {
    return {"S_CMP_EQ_I32"};
  }
  if (instruction_name == "S_CMPK_LG_I32") {
    return {"S_CMP_LG_I32"};
  }
  if (instruction_name == "S_CMPK_GT_I32") {
    return {"S_CMP_GT_I32"};
  }
  if (instruction_name == "S_CMPK_GE_I32") {
    return {"S_CMP_GE_I32"};
  }
  if (instruction_name == "S_CMPK_LT_I32") {
    return {"S_CMP_LT_I32"};
  }
  if (instruction_name == "S_CMPK_LE_I32") {
    return {"S_CMP_LE_I32"};
  }
  if (instruction_name == "S_CMPK_EQ_U32") {
    return {"S_CMP_EQ_U32"};
  }
  if (instruction_name == "S_CMPK_LG_U32") {
    return {"S_CMP_LG_U32"};
  }
  if (instruction_name == "S_CMPK_GT_U32") {
    return {"S_CMP_GT_U32"};
  }
  if (instruction_name == "S_CMPK_GE_U32") {
    return {"S_CMP_GE_U32"};
  }
  if (instruction_name == "S_CMPK_LT_U32") {
    return {"S_CMP_LT_U32"};
  }
  if (instruction_name == "S_CMPK_LE_U32") {
    return {"S_CMP_LE_U32"};
  }
  return {instruction_name};
}

std::optional<std::vector<std::uint32_t>> BuildCandidateWords(
    std::string_view instruction_name,
    const InstructionEncodingSpec& encoding) {
  const std::uint32_t opcode = encoding.opcode;
  const std::string_view encoding_name = encoding.encoding_name;

  if (encoding_name == "ENC_SOPP") {
    return std::vector<std::uint32_t>{MakeSopp(opcode)};
  }
  if (encoding_name == "ENC_SOPC") {
    return std::vector<std::uint32_t>{MakeSopc(opcode, 0, 0)};
  }
  if (encoding_name == "ENC_SOPK") {
    return std::vector<std::uint32_t>{MakeSopk(opcode, 0, 0)};
  }
  if (encoding_name == "ENC_SOP1") {
    return std::vector<std::uint32_t>{MakeSop1(opcode, 0, 0)};
  }
  if (encoding_name == "ENC_SOP2") {
    return std::vector<std::uint32_t>{MakeSop2(opcode, 0, 0, 0)};
  }
  if (encoding_name == "ENC_VOP1") {
    return std::vector<std::uint32_t>{MakeVop1(opcode, 0, 0)};
  }
  if (encoding_name == "ENC_VOP2") {
    return std::vector<std::uint32_t>{MakeVop2(opcode, 0, 0, 0)};
  }
  if (encoding_name == "ENC_VOPC") {
    return std::vector<std::uint32_t>{MakeVopc(opcode, 0, 0)};
  }
  if (encoding_name == "ENC_VOP3") {
    if (instruction_name == "V_CNDMASK_B32") {
      const auto words = MakeVop3(opcode, 0, 106, 0, 0);
      return std::vector<std::uint32_t>{words[0], words[1]};
    }
    const auto words = MakeVop3(opcode, 0, 0, 0, 0);
    return std::vector<std::uint32_t>{words[0], words[1]};
  }
  if (encoding_name == "VOP3_SDST_ENC") {
    std::uint32_t sdst = 0;
    std::uint32_t src2 = 0;
    if (IsVectorCarryOutBinaryOpcode(instruction_name)) {
      sdst = 106;
    } else if (IsVectorCarryInBinaryOpcode(instruction_name)) {
      sdst = 106;
      src2 = 106;
    }
    const auto words = MakeVop3Sdst(opcode, 0, sdst, 0, 0, src2);
    return std::vector<std::uint32_t>{words[0], words[1]};
  }
  if (encoding_name == "ENC_SMEM") {
    const auto words = MakeSmem(opcode, 0, 0, true, 0);
    return std::vector<std::uint32_t>{words[0], words[1]};
  }
  if (encoding_name == "ENC_FLAT") {
    const auto words = MakeFlat(opcode, 0, 0, 0, 0);
    return std::vector<std::uint32_t>{words[0], words[1]};
  }
  if (encoding_name == "ENC_DS") {
    const auto words = MakeDs(opcode, 0, 0, 0, 0, 0);
    return std::vector<std::uint32_t>{words[0], words[1]};
  }
  if (encoding_name == "ENC_FLAT_GLBL") {
    if (instruction_name.starts_with("GLOBAL_ATOMIC_")) {
      const auto words = MakeGlobalAtomic(opcode, false, 0, 0, 0, 0, 0);
      return std::vector<std::uint32_t>{words[0], words[1]};
    }
    const auto words = MakeGlobal(opcode, 0, 0, 0, 0, 0);
    return std::vector<std::uint32_t>{words[0], words[1]};
  }
  return std::nullopt;
}

bool IsExpectedDecodedOpcode(std::string_view catalog_name,
                             std::string_view decoded_name) {
  const std::vector<std::string_view> expected =
      ExpectedDecodedOpcodes(catalog_name);
  for (const std::string_view expected_name : expected) {
    if (expected_name == decoded_name) {
      return true;
    }
  }
  return false;
}

std::string JsonEscape(std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size() + 8);
  for (const char c : value) {
    switch (c) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += c;
        break;
    }
  }
  return escaped;
}

struct InstructionCoverage {
  std::string instruction_name;
  bool semantic_supported = false;
  bool decode_measured = false;
  bool decode_supported = false;
  std::vector<std::string> measured_encodings;
  std::vector<std::string> successful_encodings;
  std::vector<std::string> decoded_opcodes;
};

void EmitStringArray(const std::vector<std::string>& values) {
  std::cout << "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      std::cout << ",";
    }
    std::cout << "\"" << JsonEscape(values[index]) << "\"";
  }
  std::cout << "]";
}

}  // namespace

int main() {
  using namespace mirage::sim::isa;

  const InstructionCatalogMetadata& metadata = GetGfx950InstructionCatalogMetadata();
  const std::span<const InstructionSpec> instructions = GetGfx950InstructionSpecs();

  Gfx950Interpreter interpreter;
  Gfx950BinaryDecoder decoder;

  std::vector<InstructionCoverage> coverage;
  coverage.reserve(instructions.size());
  std::set<std::string> unmeasured_encodings;

  std::size_t semantic_supported_count = 0;
  std::size_t decode_supported_count = 0;
  std::size_t decode_measured_count = 0;

  for (const InstructionSpec& instruction : instructions) {
    InstructionCoverage item;
    item.instruction_name = std::string(instruction.instruction_name);
    item.semantic_supported = interpreter.Supports(instruction.instruction_name);
    if (item.semantic_supported) {
      ++semantic_supported_count;
    }

    std::set<std::string> decoded_opcodes;
    std::set<std::string> measured_encodings;
    std::set<std::string> successful_encodings;

    for (const InstructionEncodingSpec& encoding : GetEncodings(instruction)) {
      const std::optional<std::vector<std::uint32_t>> candidate =
          BuildCandidateWords(instruction.instruction_name, encoding);
      if (!candidate.has_value()) {
        unmeasured_encodings.emplace(encoding.encoding_name);
        continue;
      }

      const std::string encoding_key =
          std::string(encoding.encoding_name) + ":" +
          std::string(encoding.encoding_condition);
      measured_encodings.emplace(encoding_key);

      DecodedInstruction decoded_instruction;
      std::size_t words_consumed = 0;
      std::string error_message;
      if (!decoder.DecodeInstruction(*candidate, &decoded_instruction,
                                     &words_consumed, &error_message)) {
        continue;
      }
      if (!IsExpectedDecodedOpcode(instruction.instruction_name,
                                   decoded_instruction.opcode)) {
        continue;
      }
      item.decode_supported = true;
      successful_encodings.emplace(encoding_key);
      decoded_opcodes.emplace(std::string(decoded_instruction.opcode));
    }

    item.decode_measured = !measured_encodings.empty();
    if (item.decode_measured) {
      ++decode_measured_count;
    }
    if (item.decode_supported) {
      ++decode_supported_count;
    }

    item.measured_encodings.assign(measured_encodings.begin(),
                                   measured_encodings.end());
    item.successful_encodings.assign(successful_encodings.begin(),
                                     successful_encodings.end());
    item.decoded_opcodes.assign(decoded_opcodes.begin(), decoded_opcodes.end());
    coverage.push_back(std::move(item));
  }

  std::cout << "{\n";
  std::cout << "  \"catalog\": {\n";
  std::cout << "    \"gfx_target\": \"" << JsonEscape(metadata.gfx_target)
            << "\",\n";
  std::cout << "    \"architecture_name\": \""
            << JsonEscape(metadata.architecture_name) << "\",\n";
  std::cout << "    \"release_date\": \"" << JsonEscape(metadata.release_date)
            << "\",\n";
  std::cout << "    \"schema_version\": \""
            << JsonEscape(metadata.schema_version) << "\",\n";
  std::cout << "    \"source_xml\": \"" << JsonEscape(metadata.source_xml)
            << "\",\n";
  std::cout << "    \"instruction_count\": " << metadata.instruction_count
            << ",\n";
  std::cout << "    \"encoding_count\": " << metadata.encoding_count << "\n";
  std::cout << "  },\n";
  std::cout << "  \"summary\": {\n";
  std::cout << "    \"semantic_supported\": " << semantic_supported_count
            << ",\n";
  std::cout << "    \"decode_supported\": " << decode_supported_count << ",\n";
  std::cout << "    \"decode_measured\": " << decode_measured_count << ",\n";
  std::cout << "    \"semantic_supported_percent_total\": "
            << static_cast<double>(semantic_supported_count) /
                   static_cast<double>(metadata.instruction_count)
            << ",\n";
  std::cout << "    \"decode_supported_percent_total\": "
            << static_cast<double>(decode_supported_count) /
                   static_cast<double>(metadata.instruction_count)
            << ",\n";
  std::cout << "    \"decode_supported_percent_measured\": "
            << (decode_measured_count == 0
                    ? 0.0
                    : static_cast<double>(decode_supported_count) /
                          static_cast<double>(decode_measured_count))
            << "\n";
  std::cout << "  },\n";
  std::cout << "  \"unmeasured_encoding_families\": [";
  bool first_unmeasured = true;
  for (const std::string& encoding_name : unmeasured_encodings) {
    if (!first_unmeasured) {
      std::cout << ",";
    }
    first_unmeasured = false;
    std::cout << "\"" << JsonEscape(encoding_name) << "\"";
  }
  std::cout << "],\n";
  std::cout << "  \"instructions\": [\n";
  for (std::size_t index = 0; index < coverage.size(); ++index) {
    const InstructionCoverage& item = coverage[index];
    std::cout << "    {\n";
    std::cout << "      \"instruction_name\": \""
              << JsonEscape(item.instruction_name) << "\",\n";
    std::cout << "      \"semantic_supported\": "
              << (item.semantic_supported ? "true" : "false") << ",\n";
    std::cout << "      \"decode_measured\": "
              << (item.decode_measured ? "true" : "false") << ",\n";
    std::cout << "      \"decode_supported\": "
              << (item.decode_supported ? "true" : "false") << ",\n";
    std::cout << "      \"measured_encodings\": ";
    EmitStringArray(item.measured_encodings);
    std::cout << ",\n";
    std::cout << "      \"successful_encodings\": ";
    EmitStringArray(item.successful_encodings);
    std::cout << ",\n";
    std::cout << "      \"decoded_opcodes\": ";
    EmitStringArray(item.decoded_opcodes);
    std::cout << "\n";
    std::cout << "    }";
    if (index + 1 != coverage.size()) {
      std::cout << ",";
    }
    std::cout << "\n";
  }
  std::cout << "  ]\n";
  std::cout << "}\n";

  return 0;
}
