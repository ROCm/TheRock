#include <iostream>

#include "lib/sim/isa/instruction_catalog.h"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }
  return true;
}

}  // namespace

int main() {
  using namespace mirage::sim::isa;

  const InstructionCatalogMetadata& metadata = GetGfx950InstructionCatalogMetadata();
  if (!Expect(metadata.gfx_target == "gfx950", "expected gfx950 target") ||
      !Expect(metadata.architecture_name == "AMD CDNA 4",
              "expected AMD CDNA 4 architecture") ||
      !Expect(metadata.source_xml == "amdgpu_isa_cdna4.xml",
              "expected CDNA4 source xml") ||
      !Expect(metadata.instruction_count > 1000,
              "expected full instruction catalog") ||
      !Expect(metadata.encoding_count > metadata.instruction_count,
              "expected multiple encoding variants")) {
    return 1;
  }

  const auto instructions = GetGfx950InstructionSpecs();
  const auto encodings = GetGfx950InstructionEncodingSpecs();
  if (!Expect(instructions.size() == metadata.instruction_count,
              "expected metadata instruction count to match") ||
      !Expect(encodings.size() == metadata.encoding_count,
              "expected metadata encoding count to match")) {
    return 1;
  }

  const InstructionSpec* s_add_u32 = FindGfx950Instruction("S_ADD_U32");
  const InstructionSpec* v_add_u32 = FindGfx950Instruction("V_ADD_U32");
  const InstructionSpec* ds_add_u32 = FindGfx950Instruction("DS_ADD_U32");
  const InstructionSpec* s_endpgm = FindGfx950Instruction("S_ENDPGM");
  const InstructionSpec* missing = FindGfx950Instruction("MIRAGE_FAKE_OPCODE");

  if (!Expect(s_add_u32 != nullptr, "expected S_ADD_U32") ||
      !Expect(v_add_u32 != nullptr, "expected V_ADD_U32") ||
      !Expect(ds_add_u32 != nullptr, "expected DS_ADD_U32") ||
      !Expect(s_endpgm != nullptr, "expected S_ENDPGM") ||
      !Expect(missing == nullptr, "expected missing opcode lookup to fail")) {
    return 1;
  }

  if (!Expect(!s_add_u32->flags.is_branch, "expected S_ADD_U32 to be non-branch") ||
      !Expect(ds_add_u32->encoding_count >= 1, "expected DS_ADD_U32 encodings") ||
      !Expect(GetEncodings(*v_add_u32).size() >= 1,
              "expected V_ADD_U32 encoding list") ||
      !Expect(s_endpgm->flags.is_program_terminator,
              "expected S_ENDPGM to terminate")) {
    return 1;
  }

  return 0;
}
