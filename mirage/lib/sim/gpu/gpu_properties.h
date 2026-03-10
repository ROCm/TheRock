#ifndef MIRAGE_SIM_GPU_GPU_PROPERTIES_H_
#define MIRAGE_SIM_GPU_GPU_PROPERTIES_H_

#include <cstdint>
#include <string>

namespace mirage::sim::gpu {

struct GpuProperties {
  std::string arch_name;
  std::string gfx_target;
  std::uint32_t compute_units = 0;
  std::uint32_t simd_per_cu = 4;
  std::uint32_t wavefront_size = 64;
  std::uint64_t hbm_bytes = 0;
};

struct SignalState {
  std::uint64_t signal_id = 0;
  std::uint64_t current_value = 0;
};

}  // namespace mirage::sim::gpu

#endif  // MIRAGE_SIM_GPU_GPU_PROPERTIES_H_
