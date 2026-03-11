#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

#include "lib/sim/gpu/gpu_properties.h"
#include "lib/sim/memory/memory_region.h"
#include "lib/sim/timing/simdojo_dispatch_harness.h"

namespace {

template <typename T>
std::span<std::byte> AsWritableBytes(std::vector<T>& values) {
  return std::as_writable_bytes(std::span<T>(values.data(), values.size()));
}

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }
  return true;
}

}  // namespace

int main() {
  using namespace mirage::sim;

  gpu::GpuProperties properties;
  properties.arch_name = "CDNA4";
  properties.gfx_target = "gfx950";
  properties.compute_units = 256;
  properties.hbm_bytes = 288ULL * 1000ULL * 1000ULL * 1000ULL;

  timing::SimdojoSingleGpuHarness::Config config;
  config.dispatch_latency = 7;
  config.max_ticks = 32;

  timing::SimdojoSingleGpuHarness harness(properties, config);

  const auto dst_alloc = harness.simulator().AllocateMemory(
      memory::MemoryRegionKind::kHbm, 4 * sizeof(std::uint32_t));
  if (!Expect(dst_alloc.mapped_va != 0, "expected destination allocation")) {
    return 1;
  }

  exec::SyntheticDispatchPacket fill_dispatch;
  fill_dispatch.opcode = exec::SyntheticKernelOpcode::kFill32;
  fill_dispatch.args.dst_va = dst_alloc.mapped_va;
  fill_dispatch.args.element_count = 4;
  fill_dispatch.args.immediate_u32 = 0x12345678u;

  harness.QueueDispatch(fill_dispatch, /*send_tick=*/3);
  if (!Expect(harness.Run(), "expected simdojo timing harness run to succeed")) {
    return 1;
  }

  const auto& completions = harness.executor().completions();
  if (!Expect(completions.size() == 1,
              "expected one timed completion record") ||
      !Expect(completions.front().completion.completed,
              "expected dispatch to complete") ||
      !Expect(completions.front().completion.success,
              "expected dispatch to succeed") ||
      !Expect(completions.front().completion.dispatch_id == 1,
              "expected first dispatch identifier") ||
      !Expect(completions.front().completion_tick == 10,
              "expected completion tick to match send plus link latency")) {
    return 1;
  }

  std::vector<std::uint32_t> values(4, 0);
  if (!Expect(harness.simulator().ReadMemory(dst_alloc.mapped_va,
                                             AsWritableBytes(values)),
              "expected destination read to succeed")) {
    return 1;
  }

  constexpr std::array<std::uint32_t, 4> kExpectedValues = {
      0x12345678u,
      0x12345678u,
      0x12345678u,
      0x12345678u,
  };
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (!Expect(values[index] == kExpectedValues[index],
                "unexpected memory value after timed fill dispatch")) {
      return 1;
    }
  }

  return 0;
}
