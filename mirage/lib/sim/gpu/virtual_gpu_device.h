#ifndef MIRAGE_SIM_GPU_VIRTUAL_GPU_DEVICE_H_
#define MIRAGE_SIM_GPU_VIRTUAL_GPU_DEVICE_H_

#include <cstdint>
#include <optional>
#include <unordered_map>

#include "lib/sim/gpu/gpu_properties.h"
#include "lib/sim/queue/queue_state.h"

namespace mirage::sim::gpu {

using QueueId = queue::QueueId;

class VirtualGpuDevice {
 public:
  VirtualGpuDevice() = default;
  explicit VirtualGpuDevice(GpuProperties properties);

  void Initialize(GpuProperties properties);

  queue::QueueState CreateQueue(queue::QueueDescriptor descriptor);
  QueueId CreateComputeQueue(std::uint64_t ring_size_bytes = 4096);
  QueueId CreateDmaQueue(std::uint64_t ring_size_bytes = 4096);

  SignalState CreateSignal(std::uint64_t initial_value = 0);
  std::optional<SignalState> QuerySignal(std::uint64_t signal_id) const;
  bool WriteSignal(std::uint64_t signal_id, std::uint64_t value);

  const GpuProperties& QueryProperties() const { return properties_; }
  std::optional<queue::QueueState> QueryQueue(QueueId queue_id) const;
  bool RingDoorbell(QueueId queue_id, std::uint64_t write_ptr);
  bool RetireTo(QueueId queue_id, std::uint64_t read_ptr);

 private:
  GpuProperties properties_;
  QueueId next_queue_id_ = 1;
  std::uint64_t next_signal_id_ = 1;
  std::unordered_map<QueueId, queue::QueueState> queues_;
  std::unordered_map<std::uint64_t, SignalState> signals_;
};

}  // namespace mirage::sim::gpu

#endif  // MIRAGE_SIM_GPU_VIRTUAL_GPU_DEVICE_H_
