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
              "expected phase-1 encoding focus list")) {
    return 1;
  }

  DecodedInstruction decoded_instruction;
  std::size_t words_consumed = 99;
  std::string error_message;
  const std::array<std::uint32_t, 1> words{0u};
  if (!Expect(!decoder.DecodeInstruction(words, &decoded_instruction,
                                         &words_consumed, &error_message),
              "decoder scaffold should fail on non-empty input") ||
      !Expect(words_consumed == 0u, "expected no words consumed") ||
      !Expect(error_message.find("ENC_SOPP") != std::string::npos,
              "expected phase-0 scalar control in error") ||
      !Expect(error_message.find("ENC_VGLOBAL") != std::string::npos,
              "expected phase-0 global memory in error")) {
    return 1;
  }

  Gfx1201Interpreter interpreter;
  if (!Expect(!interpreter.Supports("S_ENDPGM"),
              "expected interpreter scaffold to reject support claims") ||
      !Expect(interpreter.CarryOverFamilyFocus().size() == 7u,
              "expected carry-over family focus list") ||
      !Expect(interpreter.Rdna4DeltaFamilyFocus().size() == 10u,
              "expected RDNA4 delta family focus list")) {
    return 1;
  }

  const std::array<DecodedInstruction, 1> program{
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  std::vector<Gfx1201CompiledInstruction> compiled_program;
  if (!Expect(!interpreter.CompileProgram(program, &compiled_program,
                                          &error_message),
              "interpreter scaffold should fail on non-empty program") ||
      !Expect(compiled_program.empty(), "expected no compiled instructions") ||
      !Expect(error_message.find("carry-over families: v, s, ds, global") !=
                  std::string::npos,
              "expected carry-over family summary") ||
      !Expect(error_message.find("image") != std::string::npos,
              "expected RDNA4 delta family summary")) {
    return 1;
  }

  WaveExecutionState state;
  if (!Expect(!interpreter.ExecuteProgram(program, &state, &error_message),
              "expected execute to fail until bring-up exists") ||
      !Expect(error_message.find("RDNA4 delta families") != std::string::npos,
              "expected execute error to describe delta families")) {
    return 1;
  }

  return 0;
}
