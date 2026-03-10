#ifndef MIRAGE_SIM_TRACE_TRACE_SINK_H_
#define MIRAGE_SIM_TRACE_TRACE_SINK_H_

#include <cstdint>

namespace mirage::sim::trace {

enum class TraceEventKind {
  kAlloc,
  kMap,
  kQueueSubmit,
  kDispatchStart,
  kDispatchComplete,
  kTransfer,
};

struct TraceEvent {
  TraceEventKind kind = TraceEventKind::kAlloc;
  std::uint64_t timestamp = 0;
  std::uint64_t object_id = 0;
};

class TraceSink {
 public:
  virtual ~TraceSink() = default;
  virtual void Emit(const TraceEvent& event) = 0;
};

}  // namespace mirage::sim::trace

#endif  // MIRAGE_SIM_TRACE_TRACE_SINK_H_
