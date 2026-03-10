// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.

#ifndef SIMDOJO_MESSAGE_H_
#define SIMDOJO_MESSAGE_H_

#include "simdojo/sim_types.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace simdojo {

/// @brief Common header for all messages sent over links.
struct MessageHeader {
  Tick timestamp = 0;        ///< Simulation tick when the message was sent.
  Tick latency = 0;          ///< Propagation delay set by the link.
  uint32_t size_bytes = 0;   ///< Payload size in bytes (for bandwidth modeling).
  PortID src_port = 0;       ///< Source port identifier.
  PortID dst_port = 0;       ///< Destination port identifier.
  uint64_t sequence_num = 0; ///< Sender-assigned sequence number.
};

/// @brief Base class for simulation messages sent over links.
///
/// @details Subclass to add payload data. The header carries routing and
/// timing metadata; arrival_tick() computes when the message reaches its
/// destination.
class Message {
public:
  /// @brief Construct a message with the given header.
  /// @param hdr The message header (routing and timing metadata).
  explicit Message(MessageHeader hdr) : header_(std::move(hdr)) {}
  virtual ~Message() = default;

  /// @brief Return the message header (const).
  /// @returns Const reference to the header.
  const MessageHeader &header() const { return header_; }

  /// @brief Return the message header (mutable).
  /// @returns Mutable reference to the header.
  MessageHeader &header() { return header_; }

  /// @brief Compute the simulation tick at which this message arrives.
  /// @returns timestamp + latency.
  Tick arrival_tick() const { return header_.timestamp + header_.latency; }

protected:
  MessageHeader header_; ///< Routing and timing metadata.
};

/// @brief A timestamp advance message used for LBTS synchronization in the
/// Chandy-Misra-Bryant protocol.
///
/// @details Carries no payload. Declares that no real message with a timestamp
/// earlier than this message's timestamp will be sent on this link, allowing
/// the receiver to safely advance its LBTS. Known in the literature as a
/// "null message" [Chandy-Misra 1979, Bryant 1977].
class TimestampAdvanceMessage final : public Message {
public:
  /// @brief Construct a timestamp advance message.
  /// @param timestamp The lower bound timestamp being declared.
  /// @param src Source port identifier.
  /// @param dst Destination port identifier.
  TimestampAdvanceMessage(Tick timestamp, PortID src, PortID dst)
      : Message(MessageHeader{.timestamp = timestamp,
                              .latency = 0,
                              .size_bytes = 0,
                              .src_port = src,
                              .dst_port = dst,
                              .sequence_num = 0}) {}
};

/// @brief A bounded priority queue of messages ordered by arrival tick
/// (min-heap).
///
/// @details Used by QueuedLink to buffer messages for explicit consumption
/// by the receiving component. Capacity is fixed at construction.
class MessageQueue {
public:
  /// @brief Construct a message queue with the given capacity.
  /// @param capacity Maximum number of messages the queue can hold.
  explicit MessageQueue(size_t capacity) : capacity_(capacity) {}

  /// @brief Try to enqueue a message.
  /// @param msg The message to enqueue (ownership transferred).
  /// @retval true Message was enqueued successfully.
  /// @retval false Queue is full; message was not enqueued.
  bool push(std::unique_ptr<Message> msg) {
    if (entries_.size() >= capacity_)
      return false;
    entries_.push_back(std::move(msg));
    std::push_heap(entries_.begin(), entries_.end(), Greater{});
    return true;
  }

  /// @brief Dequeue and return the earliest message.
  /// @returns The message with the smallest arrival tick.
  std::unique_ptr<Message> pop() {
    assert(!entries_.empty());
    std::pop_heap(entries_.begin(), entries_.end(), Greater{});
    auto msg = std::move(entries_.back());
    entries_.pop_back();
    return msg;
  }

  /// @brief Peek at the earliest message without removing it.
  /// @returns Pointer to the earliest message, or nullptr if empty.
  const Message *peek() const { return entries_.empty() ? nullptr : entries_.front().get(); }

  /// @brief Check whether the queue is empty.
  /// @retval true No messages are buffered.
  /// @retval false At least one message is buffered.
  bool empty() const { return entries_.empty(); }

  /// @brief Check whether the queue is at capacity.
  /// @retval true Queue has reached its maximum capacity.
  /// @retval false Queue has room for more messages.
  bool full() const { return entries_.size() >= capacity_; }

  /// @brief Return the number of buffered messages.
  /// @returns Current queue size.
  size_t size() const { return entries_.size(); }

  /// @brief Return the maximum number of messages the queue can hold.
  /// @returns Queue capacity.
  size_t capacity() const { return capacity_; }

  /// @brief Return the arrival tick of the earliest message.
  /// @returns Arrival tick, or TICK_MAX if empty.
  Tick next_message_time() const {
    return entries_.empty() ? TICK_MAX : entries_.front()->arrival_tick();
  }

private:
  /// @brief Min-heap comparator: smallest arrival tick first.
  struct Greater {
    bool operator()(const std::unique_ptr<Message> &a, const std::unique_ptr<Message> &b) const {
      return a->arrival_tick() > b->arrival_tick();
    }
  };

  size_t capacity_;                               ///< Maximum number of buffered messages.
  std::vector<std::unique_ptr<Message>> entries_; ///< Min-heap of messages by arrival tick.
};

} // namespace simdojo

#endif // SIMDOJO_MESSAGE_H_
