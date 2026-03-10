#ifndef MIRAGE_SIM_FABRIC_FABRIC_MODEL_H_
#define MIRAGE_SIM_FABRIC_FABRIC_MODEL_H_

#include <cstdint>
#include <string>
#include <vector>

namespace mirage::sim::fabric {

struct FabricEndpoint {
  std::string node_id;
  std::string gpu_id;
};

struct TransferRoute {
  FabricEndpoint source;
  FabricEndpoint target;
  std::uint64_t bandwidth_bytes_per_second = 0;
  std::uint64_t latency_ns = 0;
};

class FabricModel {
 public:
  void AddRoute(TransferRoute route) { routes_.push_back(route); }

  const std::vector<TransferRoute>& routes() const { return routes_; }

 private:
  std::vector<TransferRoute> routes_;
};

}  // namespace mirage::sim::fabric

#endif  // MIRAGE_SIM_FABRIC_FABRIC_MODEL_H_
