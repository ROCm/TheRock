#ifndef MIRAGE_SIM_SINGLE_GPU_SIMULATOR_H_
#define MIRAGE_SIM_SINGLE_GPU_SIMULATOR_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

#include "lib/sim/exec/dispatch/dispatch_context.h"
#include "lib/sim/isa/gfx950_interpreter.h"
#include "lib/sim/gpu/virtual_gpu_device.h"
#include "lib/sim/memory/gpu_va_space.h"
#include "lib/sim/memory/memory_region.h"
#include "lib/sim/queue/queue_state.h"

namespace mirage::sim {

class SingleGpuSimulator {
 public:
  struct DecodeCacheStats {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::size_t entries = 0;
  };

  explicit SingleGpuSimulator(gpu::GpuProperties properties);

  gpu::VirtualGpuDevice& device() { return device_; }
  const gpu::VirtualGpuDevice& device() const { return device_; }

  memory::AllocationHandle AllocateMemory(memory::MemoryRegionKind kind,
                                          std::uint64_t size_bytes,
                                          std::uint64_t alignment = 4096);
  bool WriteMemory(std::uint64_t va, std::span<const std::byte> data);
  bool ReadMemory(std::uint64_t va, std::span<std::byte> data) const;

  template <typename T>
  bool WriteObject(std::uint64_t va, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* begin = reinterpret_cast<const std::byte*>(&value);
    return WriteMemory(va, std::span<const std::byte>(begin, sizeof(T)));
  }

  template <typename T>
  std::optional<T> ReadObject(std::uint64_t va) const {
    static_assert(std::is_trivially_copyable_v<T>);
    T value{};
    auto* begin = reinterpret_cast<std::byte*>(&value);
    if (!ReadMemory(va, std::span<std::byte>(begin, sizeof(T)))) {
      return std::nullopt;
    }
    return value;
  }

  queue::QueueId CreateComputeQueue(std::uint64_t ring_size_bytes = 4096);
  std::optional<queue::QueueState> GetQueue(queue::QueueId queue_id) const;
  DecodeCacheStats GetDecodeCacheStats() const;
  exec::CompletionRecord Submit(queue::QueueId queue_id,
                                const exec::SyntheticDispatchPacket& packet);

 private:
  struct AllocationRecord;

  struct AllocationView {
    AllocationRecord* record = nullptr;
    std::byte* data = nullptr;
  };

  struct ConstAllocationView {
    const AllocationRecord* record = nullptr;
    const std::byte* data = nullptr;
  };

  struct AllocationRecord {
    memory::AllocationHandle handle;
    std::vector<std::byte> bytes;
    std::uint64_t write_version = 0;
  };

  struct DecodedProgramCacheKey {
    std::uint64_t code_va = 0;
    std::uint32_t code_word_count = 0;

    bool operator<(const DecodedProgramCacheKey& other) const {
      if (code_va != other.code_va) {
        return code_va < other.code_va;
      }
      return code_word_count < other.code_word_count;
    }
  };

  struct DecodedProgramCacheEntry {
    std::uint64_t allocation_id = 0;
    std::uint64_t allocation_write_version = 0;
    std::shared_ptr<const std::vector<isa::CompiledInstruction>> program;
  };

  AllocationRecord* FindAllocation(std::uint64_t va, std::size_t size_bytes);
  const AllocationRecord* FindAllocation(std::uint64_t va,
                                         std::size_t size_bytes) const;
  std::optional<AllocationView> FindAllocationView(std::uint64_t va,
                                                   std::size_t size_bytes);
  std::optional<ConstAllocationView> FindAllocationView(
      std::uint64_t va,
      std::size_t size_bytes) const;
  bool LoadDecodedGfx950Program(
      std::uint64_t code_va,
      std::uint32_t code_word_count,
      std::shared_ptr<const std::vector<isa::CompiledInstruction>>* program);
  exec::CompletionRecord ExecuteDispatch(
      std::uint64_t dispatch_id,
      const exec::SyntheticDispatchPacket& packet);
  bool ExecuteFill32(const exec::SyntheticDispatchPacket& packet);
  bool ExecuteVectorAddI32(const exec::SyntheticDispatchPacket& packet);
  bool ExecuteGfx950Program(const exec::SyntheticDispatchPacket& packet);

  gpu::VirtualGpuDevice device_;
  memory::GpuVaSpace va_space_;
  std::uint64_t next_allocation_id_ = 1;
  std::uint64_t next_dispatch_id_ = 1;
  std::map<std::uint64_t, AllocationRecord> allocations_;
  std::map<DecodedProgramCacheKey, DecodedProgramCacheEntry> decoded_program_cache_;
  std::uint64_t decode_cache_hits_ = 0;
  std::uint64_t decode_cache_misses_ = 0;
  std::vector<std::byte> shared_lds_scratch_;
};

}  // namespace mirage::sim

#endif  // MIRAGE_SIM_SINGLE_GPU_SIMULATOR_H_
