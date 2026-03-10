#ifndef MIRAGE_SIM_MEMORY_MEMORY_REGION_H_
#define MIRAGE_SIM_MEMORY_MEMORY_REGION_H_

#include <cstdint>

namespace mirage::sim::memory {

enum class MemoryRegionKind {
  kHbm,
  kLds,
  kScratch,
};

struct MemoryRegion {
  MemoryRegionKind kind = MemoryRegionKind::kHbm;
  std::uint64_t base_va = 0;
  std::uint64_t size_bytes = 0;
};

struct AllocationHandle {
  std::uint64_t allocation_id = 0;
  MemoryRegionKind kind = MemoryRegionKind::kHbm;
  std::uint64_t size_bytes = 0;
  std::uint64_t mapped_va = 0;
};

}  // namespace mirage::sim::memory

#endif  // MIRAGE_SIM_MEMORY_MEMORY_REGION_H_
