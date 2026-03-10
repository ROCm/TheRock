#ifndef MIRAGE_SIM_ISA_GFX1201_BINARY_DECODER_H_
#define MIRAGE_SIM_ISA_GFX1201_BINARY_DECODER_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/gfx1201/architecture_profile.h"

namespace mirage::sim::isa {

class Gfx1201BinaryDecoder {
 public:
  bool DecodeInstruction(std::span<const std::uint32_t> words,
                         DecodedInstruction* instruction,
                         std::size_t* words_consumed,
                         std::string* error_message = nullptr) const;
  bool DecodeProgram(std::span<const std::uint32_t> words,
                     std::vector<DecodedInstruction>* program,
                     std::string* error_message = nullptr) const;

  std::span<const Gfx1201EncodingFocus> Phase0EncodingFocus() const;
  std::span<const Gfx1201EncodingFocus> Phase1EncodingFocus() const;
  std::string_view BringupStatus() const;
};

}  // namespace mirage::sim::isa

#endif  // MIRAGE_SIM_ISA_GFX1201_BINARY_DECODER_H_
