// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.

#ifndef SIMDOJO_SIMULATION_H_
#define SIMDOJO_SIMULATION_H_

#include "simdojo/event_queue.h"
#include "simdojo/topology.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace simdojo {

/// @brief Per-partition simulation context.
///
/// Each worker thread manages exactly one partition. Holds the partition's
/// event queue, incoming message queues, and Lower Bound on Time Stamp
/// (LBTS) state.
struct alignas(64) PartitionContext {
  PartitionContext(PartitionID pid, uint32_t num_partitions) : partition_id(pid) {
    incoming.reserve(num_partitions);
    for (uint32_t i = 0; i < num_partitions; ++i)
      incoming.push_back(std::make_unique<CrossPartitionQueue>());
  }

  PartitionContext(PartitionContext &&) noexcept = default;
  PartitionContext &operator=(PartitionContext &&) noexcept = default;
  PartitionContext(const PartitionContext &) = delete;
  PartitionContext &operator=(const PartitionContext &) = delete;

  PartitionID partition_id;           ///< This partition's ID.
  EventQueue event_queue;             ///< Thread-local event priority queue.
  Tick local_min_outgoing = TICK_MAX; ///< Min timestamp of outgoing cross-partition events.

  /// @brief Incoming cross-partition events.
  /// One CrossPartitionQueue per source partition.
  std::vector<std::unique_ptr<CrossPartitionQueue>> incoming;

  uint64_t events_processed = 0;        ///< Total events processed by this partition.
  uint64_t timestamp_advances_sent = 0; ///< Total timestamp advances sent for LBTS.
  uint64_t sync_barriers_hit = 0;       ///< Number of LBTS barrier synchronizations.

  /// @brief Drain all incoming queues into the local event queue.
  void drain_incoming();
};

/// @brief Callback invoked by the service loop once per LBTS epoch.
using ServiceCallback = std::function<void(class SimulationEngine &)>;

/// @brief The main simulation engine driving the Parallel Discrete Event
/// Simulation (PDES).
///
/// Uses a conservative Chandy-Misra-Bryant protocol with LBTS
/// synchronization. Worker threads process events with timestamp <= current
/// LBTS, then synchronize at a barrier. A dedicated service thread on the
/// main thread computes the next LBTS and runs service callbacks (watchdog,
/// stats) while workers are blocked, then releases workers for the next
/// epoch. In single-threaded mode, the main thread runs the worker loop
/// directly with no service thread.
class SimulationEngine {
public:
  /// @brief Configuration parameters for the simulation engine.
  struct Config {
    Tick max_ticks = 1'000'000;            ///< Simulation stops when LBTS reaches this tick.
    uint32_t num_threads = 1;              ///< Number of worker threads (one per partition).
    bool enable_timestamp_advances = true; ///< Send timestamp advances for LBTS advancement.
    bool verbose = false;                  ///< Print LBTS progress to debug output.
  };

  /// @brief Construct the engine from a partitioned topology.
  /// @param topology The simulation topology (must already be partitioned).
  /// @param config Engine configuration parameters.
  SimulationEngine(Topology &topology, Config config);
  ~SimulationEngine();

  /// @brief Run the simulation to completion.
  void run();

  /// @brief Enqueue an event into the target component's partition queue.
  /// Must only be called from the thread that owns the target partition.
  /// For cross-partition delivery, use send_cross_partition() instead.
  /// @param event Reusable event descriptor.
  /// @param timestamp Simulation tick at which the event fires.
  /// @param message Optional message payload (ownership transferred).
  void schedule_event(Event *event, Tick timestamp, std::unique_ptr<Message> message = nullptr);

  /// @brief Deposit an event into another partition's cross-partition inbox.
  /// @param src_partition Source partition ID (selects the incoming queue).
  /// @param dst_partition Destination partition ID.
  /// @param event Reusable event descriptor.
  /// @param timestamp Simulation tick at which the event fires.
  /// @param message Optional message payload (ownership transferred).
  void send_cross_partition(PartitionID src_partition, PartitionID dst_partition, Event *event,
                            Tick timestamp, std::unique_ptr<Message> message = nullptr);

  /// @brief Register a callback invoked by the service thread each epoch.
  /// Callbacks run while workers are blocked at the barrier, so per-partition
  /// state is quiescent and safe to read.
  /// @param callback Function called with a reference to this engine.
  void register_service_callback(ServiceCallback callback) {
    service_callbacks_.push_back(std::move(callback));
  }

  /// @brief Return the list of all partition contexts.
  /// @returns Const reference to the contexts vector.
  const std::vector<PartitionContext> &contexts() const { return contexts_; }

  /// @brief Access a partition context by partition ID.
  /// @param pid The partition to look up.
  /// @returns Mutable reference to the PartitionContext.
  PartitionContext &context(PartitionID pid) { return contexts_[pid]; }

  /// @brief Return the current global LBTS (simulation time).
  /// @returns The global lower bound on timestamp.
  Tick global_time() const { return global_lbts_.load(std::memory_order_acquire); }

private:
  /// @brief Worker loop executed by each partition thread.
  void worker_loop(PartitionID partition_id);

  /// @brief Service loop executed by the main thread (multi-threaded only).
  /// Coordinates barrier, computes LBTS, runs service callbacks.
  void service_loop();

  /// @brief Process a single heap entry: execute its event handler if present.
  /// @param ctx The partition context that owns the event queue.
  /// @param entry The heap entry to process.
  void process_event(PartitionContext &ctx, EventQueueEntry &entry);

  /// @brief Compute the new global LBTS from all partition-local values.
  /// @returns The minimum safe timestamp across all partitions.
  Tick compute_global_lbts() const;

  /// @brief Send timestamp advances on all outgoing boundary links for a partition.
  /// @param ctx The source partition's context.
  /// @param part The source partition descriptor.
  void send_timestamp_advances(PartitionContext &ctx, const Partition &part);

  /// @brief Call initialize() on all components across all partitions.
  void initialize_components();

  /// @brief Call finalize() on all components across all partitions.
  void finalize_components();

  /// @brief Return the minimum latency among cross-partition links (lookahead).
  /// @returns Minimum cross-partition link latency, or TICK_MAX if none.
  Tick min_cross_partition_latency() const { return min_cross_latency_; }

  Topology &topology_; ///< The simulation topology.
  Config config_;      ///< Engine configuration.
  /// @brief Reusable event for timestamp advance entries (no handler, never executed).
  Event timestamp_advance_event_{nullptr, EventType::TIMESTAMP_ADVANCE};
  std::vector<PartitionContext> contexts_;         ///< Per-partition state (one per thread).
  std::vector<std::jthread> workers_;              ///< Worker threads (multi-threaded mode).
  std::vector<ServiceCallback> service_callbacks_; ///< Per-epoch service callbacks.
  std::atomic<Tick> global_lbts_{0};               ///< Current global lower bound on timestamp.
  Tick min_cross_latency_ = TICK_MAX;              ///< Minimum cross-partition link latency.
  std::atomic<bool> done_{false};                  ///< Signals simulation completion.

  std::mutex barrier_mutex_;           ///< Protects barrier state.
  std::condition_variable barrier_cv_; ///< Notifies barrier arrivals/releases.
  uint32_t barrier_count_ = 0;         ///< Workers arrived at current barrier.
  uint32_t barrier_generation_ = 0;    ///< Incremented each epoch to detect release.
};

} // namespace simdojo

#endif // SIMDOJO_SIMULATION_H_
