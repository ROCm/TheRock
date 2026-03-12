#include "lib/sim/gpu/virtual_gpu_device.h"

#include <utility>

namespace mirage::sim::gpu {

VirtualGpuDevice::VirtualGpuDevice(GpuProperties properties)
    : properties_(std::move(properties)) {
  properties_.NormalizeWavefrontSize();
}

void VirtualGpuDevice::Initialize(GpuProperties properties) {
  properties_ = std::move(properties);
  properties_.NormalizeWavefrontSize();
  next_queue_id_ = 1;
  next_signal_id_ = 1;
  queues_.clear();
  signals_.clear();
}

queue::QueueState VirtualGpuDevice::CreateQueue(queue::QueueDescriptor descriptor) {
  queue::QueueState state;
  state.queue_id = next_queue_id_++;
  state.descriptor = descriptor;
  queues_[state.queue_id] = state;
  return state;
}

QueueId VirtualGpuDevice::CreateComputeQueue(std::uint64_t ring_size_bytes) {
  queue::QueueDescriptor descriptor;
  descriptor.type = queue::QueueType::kCompute;
  descriptor.ring_size_bytes = ring_size_bytes;
  return CreateQueue(descriptor).queue_id;
}

QueueId VirtualGpuDevice::CreateDmaQueue(std::uint64_t ring_size_bytes) {
  queue::QueueDescriptor descriptor;
  descriptor.type = queue::QueueType::kDma;
  descriptor.ring_size_bytes = ring_size_bytes;
  return CreateQueue(descriptor).queue_id;
}

SignalState VirtualGpuDevice::CreateSignal(std::uint64_t initial_value) {
  SignalState signal{next_signal_id_++, initial_value};
  signals_[signal.signal_id] = signal;
  return signal;
}

std::optional<SignalState> VirtualGpuDevice::QuerySignal(
    std::uint64_t signal_id) const {
  const auto it = signals_.find(signal_id);
  if (it == signals_.end()) {
    return std::nullopt;
  }
  return it->second;
}

bool VirtualGpuDevice::WriteSignal(std::uint64_t signal_id, std::uint64_t value) {
  const auto it = signals_.find(signal_id);
  if (it == signals_.end()) {
    return false;
  }
  it->second.current_value = value;
  return true;
}

std::optional<queue::QueueState> VirtualGpuDevice::QueryQueue(
    QueueId queue_id) const {
  const auto it = queues_.find(queue_id);
  if (it == queues_.end()) {
    return std::nullopt;
  }
  return it->second;
}

bool VirtualGpuDevice::RingDoorbell(QueueId queue_id, std::uint64_t write_ptr) {
  const auto it = queues_.find(queue_id);
  if (it == queues_.end()) {
    return false;
  }
  it->second.doorbell.write_ptr = write_ptr;
  return true;
}

bool VirtualGpuDevice::RetireTo(QueueId queue_id, std::uint64_t read_ptr) {
  const auto it = queues_.find(queue_id);
  if (it == queues_.end()) {
    return false;
  }
  if (read_ptr > it->second.doorbell.write_ptr) {
    return false;
  }
  it->second.doorbell.read_ptr = read_ptr;
  return true;
}

}  // namespace mirage::sim::gpu
