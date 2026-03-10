#ifndef MIRAGE_SIM_MEMORY_GPU_VA_SPACE_H_
#define MIRAGE_SIM_MEMORY_GPU_VA_SPACE_H_

#include <cstdint>

namespace mirage::sim::memory {

struct GpuVaRange {
  std::uint64_t base_va = 0;
  std::uint64_t size_bytes = 0;
};

class GpuVaSpace {
 public:
  explicit GpuVaSpace(std::uint64_t start_va = 0x100000000ULL)
      : next_va_(start_va) {}

  GpuVaRange Reserve(std::uint64_t size_bytes, std::uint64_t alignment) {
    std::uint64_t aligned = AlignUp(next_va_, alignment);
    GpuVaRange range{aligned, size_bytes};
    next_va_ = aligned + size_bytes;
    return range;
  }

 private:
  static std::uint64_t AlignUp(std::uint64_t value, std::uint64_t alignment) {
    if (alignment == 0) {
      return value;
    }
    const std::uint64_t remainder = value % alignment;
    if (remainder == 0) {
      return value;
    }
    return value + (alignment - remainder);
  }

  std::uint64_t next_va_;
};

}  // namespace mirage::sim::memory

#endif  // MIRAGE_SIM_MEMORY_GPU_VA_SPACE_H_
