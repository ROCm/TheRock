#ifndef MIRAGE_SIM_ISA_COMMON_WAVE_EXECUTION_STATE_H_
#define MIRAGE_SIM_ISA_COMMON_WAVE_EXECUTION_STATE_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include "lib/sim/isa/common/wavefront_size.h"

namespace mirage::sim::isa {

struct WaveExecutionState {
  static constexpr std::size_t kLaneCount = kWavefrontSize64;
  static constexpr std::size_t kScalarRegisterCount = 128;
  static constexpr std::size_t kVectorRegisterCount = 128;
  static constexpr std::size_t kLdsSizeBytes = 64 * 1024;

  std::array<std::uint32_t, kScalarRegisterCount> sgprs{};
  std::array<std::array<std::uint32_t, kLaneCount>, kVectorRegisterCount> vgprs{};
  std::array<std::byte, kLdsSizeBytes> lds_bytes{};
  std::uint32_t lane_count = kWavefrontSize64;
  std::uint64_t exec_mask = MaskForWavefrontSize(kWavefrontSize64);
  std::uint64_t vcc_mask = 0;
  std::uint64_t pc = 0;
  std::uint32_t workgroup_wave_count = 1;
  bool halted = false;
  bool waiting_on_barrier = false;
  bool scc = false;

  std::size_t ActiveLaneCount() const { return lane_count; }

  std::uint64_t LaneMask() const { return MaskForWavefrontSize(lane_count); }

  void SetLaneCount(std::uint32_t requested_lane_count) {
    if (requested_lane_count == 0 || requested_lane_count > kLaneCount) {
      lane_count = kWavefrontSize64;
    } else {
      lane_count = requested_lane_count;
    }
    ClampMasksToLaneCount();
  }

  void ClampMasksToLaneCount() {
    const std::uint64_t lane_mask = LaneMask();
    exec_mask &= lane_mask;
    vcc_mask &= lane_mask;
  }
};

}  // namespace mirage::sim::isa

#endif  // MIRAGE_SIM_ISA_COMMON_WAVE_EXECUTION_STATE_H_
