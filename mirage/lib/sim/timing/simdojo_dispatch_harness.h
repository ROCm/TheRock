#ifndef MIRAGE_SIM_TIMING_SIMDOJO_DISPATCH_HARNESS_H_
#define MIRAGE_SIM_TIMING_SIMDOJO_DISPATCH_HARNESS_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "lib/sim/exec/dispatch/dispatch_context.h"
#include "lib/sim/gpu/gpu_properties.h"
#include "lib/sim/queue/queue_state.h"
#include "lib/sim/single_gpu_simulator.h"
#include "simdojo/component.h"
#include "simdojo/message.h"
#include "simdojo/simulation.h"
#include "simdojo/topology.h"

namespace mirage::sim::timing {

struct TimedDispatchCompletion {
  simdojo::Tick completion_tick = 0;
  exec::CompletionRecord completion;
};

class SimdojoDispatchMessage final : public simdojo::Message {
 public:
  SimdojoDispatchMessage(simdojo::Tick send_tick,
                         exec::SyntheticDispatchPacket packet);

  const exec::SyntheticDispatchPacket& packet() const { return packet_; }

 private:
  exec::SyntheticDispatchPacket packet_{};
};

class SimdojoDispatchSource final : public simdojo::Component {
 public:
  struct PendingDispatch {
    simdojo::Tick send_tick = 0;
    exec::SyntheticDispatchPacket packet;
  };

  explicit SimdojoDispatchSource(std::string name);

  simdojo::Port* dispatch_out_port() const { return dispatch_out_port_; }
  void QueueDispatch(exec::SyntheticDispatchPacket packet,
                     simdojo::Tick send_tick = 0);
  std::size_t pending_dispatch_count() const { return pending_dispatches_.size(); }
  void initialize() override;

 private:
  simdojo::Port* dispatch_out_port_ = nullptr;
  std::vector<PendingDispatch> pending_dispatches_;
};

class SimdojoSingleGpuExecutor final : public simdojo::Component {
 public:
  SimdojoSingleGpuExecutor(std::string name,
                           SingleGpuSimulator* simulator,
                           std::uint64_t ring_size_bytes = 4096);

  simdojo::Port* dispatch_in_port() const { return dispatch_in_port_; }
  queue::QueueId queue_id() const { return queue_id_; }
  const std::vector<TimedDispatchCompletion>& completions() const {
    return completions_;
  }

  void initialize() override;

 private:
  void HandleDispatch(simdojo::Tick now, simdojo::Message* message);

  SingleGpuSimulator* simulator_ = nullptr;
  simdojo::Port* dispatch_in_port_ = nullptr;
  std::uint64_t ring_size_bytes_ = 4096;
  queue::QueueId queue_id_ = 0;
  std::vector<TimedDispatchCompletion> completions_;
};

class SimdojoSingleGpuHarness {
 public:
  struct Config {
    simdojo::Tick dispatch_latency = 1;
    simdojo::Tick max_ticks = 64;
    std::uint32_t num_threads = 1;
    std::uint64_t ring_size_bytes = 4096;
  };

  explicit SimdojoSingleGpuHarness(gpu::GpuProperties properties);
  SimdojoSingleGpuHarness(gpu::GpuProperties properties, Config config);

  SingleGpuSimulator& simulator() { return simulator_; }
  const SingleGpuSimulator& simulator() const { return simulator_; }
  SimdojoDispatchSource& source() { return *source_; }
  const SimdojoDispatchSource& source() const { return *source_; }
  SimdojoSingleGpuExecutor& executor() { return *executor_; }
  const SimdojoSingleGpuExecutor& executor() const { return *executor_; }

  void QueueDispatch(exec::SyntheticDispatchPacket packet,
                     simdojo::Tick send_tick = 0);
  bool Run();

 private:
  Config config_{};
  SingleGpuSimulator simulator_;
  simdojo::Topology topology_;
  SimdojoDispatchSource* source_ = nullptr;
  SimdojoSingleGpuExecutor* executor_ = nullptr;
  std::size_t queued_dispatch_count_ = 0;
};

}  // namespace mirage::sim::timing

#endif  // MIRAGE_SIM_TIMING_SIMDOJO_DISPATCH_HARNESS_H_
