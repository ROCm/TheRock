#ifndef MIRAGE_SIM_ISA_INSTRUCTION_CATALOG_H_
#define MIRAGE_SIM_ISA_INSTRUCTION_CATALOG_H_

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace mirage::sim::isa {

struct InstructionFlags {
  bool is_branch = false;
  bool is_conditional_branch = false;
  bool is_indirect_branch = false;
  bool is_program_terminator = false;
  bool is_immediately_executed = false;
};

struct InstructionEncodingSpec {
  std::string_view encoding_name;
  std::string_view encoding_condition;
  std::uint32_t opcode = 0;
  std::uint16_t operand_count = 0;
};

struct InstructionSpec {
  std::string_view instruction_name;
  InstructionFlags flags;
  std::uint32_t encoding_begin = 0;
  std::uint32_t encoding_count = 0;
};

struct InstructionCatalogMetadata {
  std::string_view gfx_target;
  std::string_view architecture_name;
  std::string_view release_date;
  std::string_view schema_version;
  std::string_view source_xml;
  std::uint32_t instruction_count = 0;
  std::uint32_t encoding_count = 0;
};

const InstructionCatalogMetadata& GetGfx950InstructionCatalogMetadata();
std::span<const InstructionSpec> GetGfx950InstructionSpecs();
std::span<const InstructionEncodingSpec> GetGfx950InstructionEncodingSpecs();
const InstructionSpec* FindGfx950Instruction(std::string_view instruction_name);
std::span<const InstructionEncodingSpec> GetEncodings(
    const InstructionSpec& instruction);

}  // namespace mirage::sim::isa

#endif  // MIRAGE_SIM_ISA_INSTRUCTION_CATALOG_H_
