#include <iostream>
#include <memory>

#include "simdojo/simulation.h"
#include "simdojo/topology.h"

namespace {

struct LifecycleState {
  bool initialized = false;
  bool finalized = false;
};

class TestComponent final : public simdojo::Component {
 public:
  explicit TestComponent(LifecycleState* state)
      : simdojo::Component("leaf"), state_(state) {}

  void initialize() override {
    if (state_ != nullptr) {
      state_->initialized = true;
    }
  }

  void finalize() override {
    if (state_ != nullptr) {
      state_->finalized = true;
    }
  }

 private:
  LifecycleState* state_ = nullptr;
};

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }
  return true;
}

}  // namespace

int main() {
  LifecycleState state;

  simdojo::Topology topology;
  auto root = std::make_unique<simdojo::CompositeComponent>("soc");
  TestComponent* leaf =
      static_cast<TestComponent*>(root->add_child(std::make_unique<TestComponent>(&state)));
  topology.set_root(std::move(root));
  topology.partition(1);

  simdojo::SimulationEngine::Config config;
  config.max_ticks = 1;
  config.num_threads = 1;

  simdojo::SimulationEngine engine(topology, config);
  engine.run();

  return Expect(leaf != nullptr, "leaf component was not created") &&
                 Expect(state.initialized, "simdojo component initialize() was not called") &&
                 Expect(state.finalized, "simdojo component finalize() was not called")
             ? 0
             : 1;
}
