#ifndef MIRAGE_SIM_REPLAY_REPLAY_LOG_H_
#define MIRAGE_SIM_REPLAY_REPLAY_LOG_H_

#include <cstdint>
#include <vector>

namespace mirage::sim::replay {

struct ReplayRecord {
  std::uint64_t checkpoint_id = 0;
  std::uint64_t timestamp = 0;
};

class ReplayLog {
 public:
  void AddRecord(ReplayRecord record) { records_.push_back(record); }

  const std::vector<ReplayRecord>& records() const { return records_; }

 private:
  std::vector<ReplayRecord> records_;
};

}  // namespace mirage::sim::replay

#endif  // MIRAGE_SIM_REPLAY_REPLAY_LOG_H_
