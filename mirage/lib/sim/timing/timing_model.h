#ifndef MIRAGE_SIM_TIMING_TIMING_MODEL_H_
#define MIRAGE_SIM_TIMING_TIMING_MODEL_H_

#include <cstdint>
#include <vector>

namespace mirage::sim::timing {

enum class TimingMode {
  kFunctional,
  kAnalytical,
  kDetailed,
};

struct ScheduledEvent {
  std::uint64_t timestamp = 0;
  std::uint64_t event_id = 0;
};

class EventScheduler {
 public:
  void Schedule(ScheduledEvent event) { events_.push_back(event); }

  const std::vector<ScheduledEvent>& events() const { return events_; }

 private:
  std::vector<ScheduledEvent> events_;
};

}  // namespace mirage::sim::timing

#endif  // MIRAGE_SIM_TIMING_TIMING_MODEL_H_
