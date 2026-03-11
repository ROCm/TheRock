#include "lib/sim/timing/simdojo_dispatch_harness.h"

#include <utility>

namespace mirage::sim::timing {

SimdojoSingleGpuHarness::SimdojoSingleGpuHarness(gpu::GpuProperties properties)
    : SimdojoSingleGpuHarness(std::move(properties), Config{}) {}

SimdojoDispatchMessage::SimdojoDispatchMessage(
    simdojo::Tick send_tick,
    exec::SyntheticDispatchPacket packet)
    : simdojo::Message(simdojo::MessageHeader{
          .timestamp = send_tick,
          .latency = 0,
          .size_bytes =
              static_cast<std::uint32_t>(sizeof(exec::SyntheticDispatchPacket)),
          .src_port = 0,
          .dst_port = 0,
          .sequence_num = 0,
      }),
      packet_(packet) {}

SimdojoDispatchSource::SimdojoDispatchSource(std::string name)
    : simdojo::Component(std::move(name)) {
  dispatch_out_port_ =
      add_port(std::make_unique<simdojo::Port>("dispatch_out", 0, this));
}

void SimdojoDispatchSource::QueueDispatch(exec::SyntheticDispatchPacket packet,
                                          simdojo::Tick send_tick) {
  pending_dispatches_.push_back(PendingDispatch{
      .send_tick = send_tick,
      .packet = packet,
  });
}

void SimdojoDispatchSource::initialize() {
  if (dispatch_out_port_ == nullptr || dispatch_out_port_->link() == nullptr) {
    pending_dispatches_.clear();
    return;
  }

  for (const PendingDispatch& pending : pending_dispatches_) {
    dispatch_out_port_->send(
        std::make_unique<SimdojoDispatchMessage>(pending.send_tick,
                                                 pending.packet));
  }
  pending_dispatches_.clear();
}

SimdojoSingleGpuExecutor::SimdojoSingleGpuExecutor(std::string name,
                                                   SingleGpuSimulator* simulator,
                                                   std::uint64_t ring_size_bytes)
    : simdojo::Component(std::move(name)),
      simulator_(simulator),
      ring_size_bytes_(ring_size_bytes) {
  dispatch_in_port_ =
      add_port(std::make_unique<simdojo::Port>("dispatch_in", 0, this));
  dispatch_in_port_->set_handler([this](simdojo::Tick now, simdojo::Message* message) {
    HandleDispatch(now, message);
  });
}

void SimdojoSingleGpuExecutor::initialize() {
  completions_.clear();
  if (simulator_ != nullptr && queue_id_ == 0) {
    queue_id_ = simulator_->CreateComputeQueue(ring_size_bytes_);
  }
}

void SimdojoSingleGpuExecutor::HandleDispatch(simdojo::Tick now,
                                              simdojo::Message* message) {
  if (simulator_ == nullptr || message == nullptr || queue_id_ == 0) {
    return;
  }

  const auto* dispatch_message =
      dynamic_cast<SimdojoDispatchMessage*>(message);
  if (dispatch_message == nullptr) {
    return;
  }

  exec::SyntheticDispatchPacket packet = dispatch_message->packet();
  packet.context.queue_id = queue_id_;

  TimedDispatchCompletion completion;
  completion.completion_tick = now;
  completion.completion = simulator_->Submit(queue_id_, packet);
  completions_.push_back(completion);
}

SimdojoSingleGpuHarness::SimdojoSingleGpuHarness(gpu::GpuProperties properties,
                                                 Config config)
    : config_(config), simulator_(std::move(properties)) {
  auto root = std::make_unique<simdojo::CompositeComponent>("mirage");
  source_ = static_cast<SimdojoDispatchSource*>(
      root->add_child(std::make_unique<SimdojoDispatchSource>("dispatch_source")));
  executor_ = static_cast<SimdojoSingleGpuExecutor*>(root->add_child(
      std::make_unique<SimdojoSingleGpuExecutor>("gpu_executor", &simulator_,
                                                 config_.ring_size_bytes)));

  topology_.add_link(source_->dispatch_out_port(), executor_->dispatch_in_port(),
                     config_.dispatch_latency);
  topology_.set_root(std::move(root));
  topology_.partition(config_.num_threads);
}

void SimdojoSingleGpuHarness::QueueDispatch(exec::SyntheticDispatchPacket packet,
                                            simdojo::Tick send_tick) {
  source_->QueueDispatch(packet, send_tick);
  ++queued_dispatch_count_;
}

bool SimdojoSingleGpuHarness::Run() {
  const std::size_t expected_dispatch_count = queued_dispatch_count_;
  queued_dispatch_count_ = 0;

  simdojo::SimulationEngine::Config engine_config;
  engine_config.max_ticks = config_.max_ticks;
  engine_config.num_threads = config_.num_threads;

  simdojo::SimulationEngine engine(topology_, engine_config);
  engine.run();
  return executor_->completions().size() == expected_dispatch_count;
}

}  // namespace mirage::sim::timing
