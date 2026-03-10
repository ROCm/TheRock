#include "lib/sim/isa/gfx1201/binary_decoder.h"

#include <string>

namespace mirage::sim::isa {
namespace {

std::string BuildDecoderBringupMessage() {
  std::string message =
      "gfx1201 decoder scaffold only; phase-0 compute seed encodings:";
  bool first = true;
  for (const Gfx1201DecoderSeedEncoding& encoding :
       GetGfx1201Phase0ComputeDecoderSeeds()) {
    message.append(first ? " " : ", ");
    message.append(encoding.encoding_name);
    first = false;
  }
  return message;
}

}  // namespace

bool Gfx1201BinaryDecoder::DecodeInstruction(
    std::span<const std::uint32_t> words,
    DecodedInstruction* instruction,
    std::size_t* words_consumed,
    std::string* error_message) const {
  if (instruction == nullptr || words_consumed == nullptr) {
    if (error_message != nullptr) {
      *error_message = "decode outputs must not be null";
    }
    return false;
  }

  *instruction = DecodedInstruction{};
  *words_consumed = 0;

  if (words.empty()) {
    if (error_message != nullptr) {
      *error_message = "instruction stream is empty";
    }
    return false;
  }

  if (error_message != nullptr) {
    *error_message = BuildDecoderBringupMessage();
  }
  return false;
}

bool Gfx1201BinaryDecoder::DecodeProgram(std::span<const std::uint32_t> words,
                                         std::vector<DecodedInstruction>* program,
                                         std::string* error_message) const {
  if (program == nullptr) {
    if (error_message != nullptr) {
      *error_message = "decoded program output must not be null";
    }
    return false;
  }

  program->clear();
  if (words.empty()) {
    return true;
  }

  if (error_message != nullptr) {
    *error_message = BuildDecoderBringupMessage();
  }
  return false;
}

std::span<const Gfx1201DecoderSeedEncoding>
Gfx1201BinaryDecoder::Phase0ComputeSeeds() const {
  return GetGfx1201Phase0ComputeDecoderSeeds();
}

const Gfx1201DecoderSeedEncoding* Gfx1201BinaryDecoder::FindPhase0ComputeSeed(
    std::string_view encoding_name) const {
  return FindGfx1201Phase0ComputeDecoderSeed(encoding_name);
}

std::span<const Gfx1201EncodingFocus> Gfx1201BinaryDecoder::Phase0EncodingFocus()
    const {
  return GetGfx1201Phase0DecoderFocus();
}

std::span<const Gfx1201EncodingFocus> Gfx1201BinaryDecoder::Phase1EncodingFocus()
    const {
  return GetGfx1201Phase1DecoderFocus();
}

std::string_view Gfx1201BinaryDecoder::BringupStatus() const {
  return DescribeGfx1201BringupPhase();
}

}  // namespace mirage::sim::isa
