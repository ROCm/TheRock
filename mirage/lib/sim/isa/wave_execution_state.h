#ifndef MIRAGE_SIM_ISA_WAVE_EXECUTION_STATE_H_
#define MIRAGE_SIM_ISA_WAVE_EXECUTION_STATE_H_

#include <array>
#include <cstddef>
#include <cstdint>

namespace mirage::sim::isa {

struct WaveExecutionState {
  static constexpr std::size_t kLaneCount = 64;
  static constexpr std::size_t kScalarRegisterCount = 128;
  static constexpr std::size_t kVectorRegisterCount = 128;
  static constexpr std::size_t kLdsSizeBytes = 64 * 1024;

  std::array<std::uint32_t, kScalarRegisterCount> sgprs{};
  std::array<std::array<std::uint32_t, kLaneCount>, kVectorRegisterCount> vgprs{};
  std::array<std::byte, kLdsSizeBytes> lds_bytes{};
  std::uint64_t exec_mask = ~0ULL;
  std::uint64_t vcc_mask = 0;
  std::uint64_t pc = 0;
  std::uint32_t workgroup_wave_count = 1;
  bool halted = false;
  bool waiting_on_barrier = false;
  bool scc = false;
};

}  // namespace mirage::sim::isa

#endif  // MIRAGE_SIM_ISA_WAVE_EXECUTION_STATE_H_
