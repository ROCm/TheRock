// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.

#include "simdojo/simulation.h"

#include "simdojo/debug_print.h"

#include <algorithm>
#include <cassert>

namespace simdojo {

void PartitionContext::drain_incoming() {
  for (auto &queue : incoming)
    queue->drain_into(event_queue);
}

SimulationEngine::SimulationEngine(Topology &topology, Config config)
    : topology_(topology), config_(config) {

  uint32_t num_threads = config_.num_threads;
  assert(num_threads > 0);
  assert(topology_.partitions().size() == num_threads);

  contexts_.reserve(num_threads);
  for (uint32_t i = 0; i < num_threads; ++i)
    contexts_.emplace_back(i, num_threads);

  // Compute minimum cross-partition link latency (lookahead).
  for (auto &link : topology_.links()) {
    if (link->is_cross_partition()) {
      assert(link->latency() > 0 && "cross-partition links require positive latency for LBTS");
      min_cross_latency_ = std::min(min_cross_latency_, link->latency());
    }
  }

  // Set engine pointer on all components.
  for (auto &part : topology_.partitions()) {
    for (auto *comp : part.components)
      comp->set_engine(this);
  }
}

SimulationEngine::~SimulationEngine() {
  done_.store(true, std::memory_order_release);
  barrier_cv_.notify_all();
  // jthreads auto-join on destruction.
}

void SimulationEngine::run() {
  initialize_components();

  done_.store(false, std::memory_order_release);
  global_lbts_.store(0, std::memory_order_release);
  barrier_count_ = 0;
  barrier_generation_ = 0;

  uint32_t num_threads = config_.num_threads;

  if (num_threads == 1) {
    // Single-threaded: run worker directly, no service thread needed.
    worker_loop(0);
  } else {
    // Launch worker threads for all partitions (0..N-1).
    for (uint32_t i = 0; i < num_threads; ++i)
      workers_.emplace_back([this, i]() { worker_loop(i); });

    // Main thread runs the service loop (coordination, callbacks).
    service_loop();

    // Wait for all workers.
    workers_.clear();
  }

  finalize_components();
}

void SimulationEngine::service_loop() {
  uint32_t num_threads = config_.num_threads;

  while (!done_.load(std::memory_order_acquire)) {
    // Wait for all workers to arrive at the barrier.
    {
      std::unique_lock<std::mutex> lock(barrier_mutex_);
      barrier_cv_.wait(lock, [this, num_threads] {
        return barrier_count_ == num_threads || done_.load(std::memory_order_acquire);
      });

      if (done_.load(std::memory_order_acquire))
        break;

      // Compute new LBTS.
      Tick new_lbts = compute_global_lbts();
      if (new_lbts >= config_.max_ticks)
        done_.store(true, std::memory_order_release);
      else
        global_lbts_.store(new_lbts, std::memory_order_release);

      if (done_.load(std::memory_order_acquire)) {
        barrier_count_ = 0;
        barrier_generation_++;
        barrier_cv_.notify_all();
        break;
      }

      // Run service callbacks while workers are still blocked at the barrier,
      // so per-partition state is quiescent and safe to read.
      for (auto &cb : service_callbacks_)
        cb(*this);

      if (config_.verbose) {
        debug::print("LBTS advanced to ", global_lbts_.load(std::memory_order_acquire));
      }

      // Release workers for the next epoch.
      barrier_count_ = 0;
      barrier_generation_++;
      barrier_cv_.notify_all();
    }
  }
}

void SimulationEngine::worker_loop(PartitionID partition_id) {
  PartitionContext &ctx = contexts_[partition_id];
  const Partition &part = topology_.partitions()[partition_id];
  uint32_t num_threads = config_.num_threads;

  while (!done_.load(std::memory_order_acquire)) {
    Tick current_lbts = global_lbts_.load(std::memory_order_acquire);

    // Process all events with timestamp <= current LBTS.
    while (!ctx.event_queue.empty() && ctx.event_queue.next_event_time() <= current_lbts) {
      auto entry = ctx.event_queue.pop();
      process_event(ctx, entry);
    }

    // Send timestamp advances on quiescent cross-partition links.
    if (config_.enable_timestamp_advances && num_threads > 1)
      send_timestamp_advances(ctx, part);

    ctx.sync_barriers_hit++;

    if (num_threads == 1) {
      // Single-threaded: compute LBTS directly, no barrier.
      Tick new_lbts = compute_global_lbts();
      if (new_lbts >= config_.max_ticks) {
        done_.store(true, std::memory_order_release);
        break;
      }
      global_lbts_.store(new_lbts, std::memory_order_release);

      // Run service callbacks and verbose logging (mirroring the service loop).
      for (auto &cb : service_callbacks_)
        cb(*this);

      if (config_.verbose) {
        debug::print("LBTS advanced to ", global_lbts_.load(std::memory_order_acquire));
      }

      ctx.drain_incoming();
      ctx.local_min_outgoing = TICK_MAX;
    } else {
      // Multi-threaded: signal service thread and wait for next epoch.
      {
        std::unique_lock<std::mutex> lock(barrier_mutex_);
        barrier_count_++;
        if (barrier_count_ == num_threads) {
          // Last worker: drain all partitions while everyone is blocked.
          // No producers are active, so plain vector access is safe.
          for (auto &c : contexts_) {
            c.drain_incoming();
            c.local_min_outgoing = TICK_MAX;
          }
          barrier_cv_.notify_all();
        }

        uint32_t gen = barrier_generation_;
        barrier_cv_.wait(lock, [this, gen] {
          return barrier_generation_ != gen || done_.load(std::memory_order_acquire);
        });
      }

      if (done_.load(std::memory_order_acquire))
        break;
    }
  }
}

void SimulationEngine::process_event(PartitionContext &ctx, EventQueueEntry &entry) {
  ctx.event_queue.set_current_tick(entry.timestamp);

  if (entry.event->has_handler()) {
    entry.event->execute(entry.timestamp, entry.message.get());
    ctx.events_processed++;
  }
}

void SimulationEngine::schedule_event(Event *event, Tick timestamp,
                                      std::unique_ptr<Message> message) {
  Component *target = event->target();
  assert(target != nullptr);
  PartitionID pid = target->partition_id();
  assert(pid < contexts_.size());
  contexts_[pid].event_queue.push(EventQueueEntry{timestamp, 0, event, std::move(message)});
}

void SimulationEngine::send_cross_partition(PartitionID src_partition, PartitionID dst_partition,
                                            Event *event, Tick timestamp,
                                            std::unique_ptr<Message> message) {
  assert(src_partition < contexts_.size());
  assert(dst_partition < contexts_.size());

  // Deposit into the destination partition's incoming queue for this source.
  contexts_[dst_partition].incoming[src_partition]->push(
      EventQueueEntry{timestamp, 0, event, std::move(message)});

  // Update the source partition's min_outgoing.
  PartitionContext &src_ctx = contexts_[src_partition];
  if (timestamp < src_ctx.local_min_outgoing)
    src_ctx.local_min_outgoing = timestamp;
}

Tick SimulationEngine::compute_global_lbts() const {
  Tick min_time = TICK_MAX;
  for (auto &ctx : contexts_) {
    min_time = std::min(min_time, ctx.event_queue.next_event_time());
    min_time = std::min(min_time, ctx.local_min_outgoing);
  }

  // Ensure we advance by at least some amount to avoid stalling.
  Tick current = global_lbts_.load(std::memory_order_acquire);
  if (min_time <= current && min_cross_latency_ != TICK_MAX)
    min_time = current + 1;

  return min_time;
}

void SimulationEngine::send_timestamp_advances(PartitionContext &ctx, const Partition &part) {
  Tick current_lbts = global_lbts_.load(std::memory_order_acquire);

  for (auto *link : part.boundary_links) {
    if (link->src()->owner()->partition_id() != ctx.partition_id)
      continue;

    Tick advance_time = current_lbts + link->latency();
    PortID src_pid = link->src()->port_id();
    PortID dst_pid = link->dst()->port_id();

    auto msg = std::make_unique<TimestampAdvanceMessage>(advance_time, src_pid, dst_pid);
    PartitionID dst_part = link->dst()->owner()->partition_id();
    contexts_[dst_part].incoming[ctx.partition_id]->push(
        EventQueueEntry{advance_time, 0, &timestamp_advance_event_, std::move(msg)});
    ctx.timestamp_advances_sent++;
  }
}

void SimulationEngine::initialize_components() {
  for (auto &part : topology_.partitions()) {
    for (auto *comp : part.components)
      comp->initialize();
  }
}

void SimulationEngine::finalize_components() {
  for (auto &part : topology_.partitions()) {
    for (auto *comp : part.components)
      comp->finalize();
  }
}

} // namespace simdojo
