#ifndef MIRAGE_SIM_QUEUE_QUEUE_STATE_H_
#define MIRAGE_SIM_QUEUE_QUEUE_STATE_H_

#include <cstdint>
#include <vector>

namespace mirage::sim::queue {

using QueueId = std::uint64_t;

enum class QueueType {
  kCompute,
  kDma,
};

struct QueueDescriptor {
  QueueType type = QueueType::kCompute;
  std::uint64_t ring_size_bytes = 0;
};

struct PacketStream {
  std::vector<std::uint32_t> dwords;
};

struct DoorbellState {
  std::uint64_t write_ptr = 0;
  std::uint64_t read_ptr = 0;
};

struct QueueState {
  QueueId queue_id = 0;
  QueueDescriptor descriptor;
  DoorbellState doorbell;
};

}  // namespace mirage::sim::queue

#endif  // MIRAGE_SIM_QUEUE_QUEUE_STATE_H_
