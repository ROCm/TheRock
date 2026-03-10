// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.

#ifndef SIMDOJO_COMPONENT_H_
#define SIMDOJO_COMPONENT_H_

#include "simdojo/event_queue.h"
#include "simdojo/message.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace simdojo {

class Port;
class Link;
class SimulationEngine;

/// @brief Base node in the simulation compound graph.
///
/// @details Nodes form a compound graph where edges are either:
///   - Inclusion edges (parent-child in the component tree)
///   - Adjacency edges (links between ports)
class Node {
public:
  /// @brief Construct a node with the given name.
  /// @param[in] name Human-readable name for this node.
  explicit Node(std::string name) : name_(std::move(name)) {}
  virtual ~Node() = default;

  /// @brief Return the human-readable name.
  /// @returns Reference to the node's name string.
  const std::string &name() const { return name_; }

  /// @brief Return the auto-assigned unique identifier.
  /// @returns The node's ComponentID.
  ComponentID id() const { return id_; }

  /// @brief Return the parent node in the component tree.
  /// @returns Pointer to the parent, or nullptr if this is the root.
  Node *parent() const { return parent_; }

  /// @brief Set the parent node.
  /// @param[in] parent Pointer to the new parent node.
  void set_parent(Node *parent) { parent_ = parent; }

  /// @brief Return the depth in the component tree (root = 0).
  /// @returns Depth as an unsigned integer.
  uint32_t depth() const { return depth_; }

  /// @brief Set the depth in the component tree.
  /// @param[in] d New depth value.
  void set_depth(uint32_t d) { depth_ = d; }

  /// @brief Full hierarchical path name (e.g., "soc.gpu.cu0.alu").
  /// @returns Dot-separated path from the root to this node.
  std::string full_path() const {
    if (parent_ == nullptr)
      return name_;
    return parent_->full_path() + "." + name_;
  }

private:
  /// @brief Global counter for auto-assigning unique IDs.
  static inline std::atomic<ComponentID> next_id_ = 0;

  const std::string name_; ///< Human-readable node name.
  const ComponentID id_ =
      next_id_.fetch_add(1, std::memory_order_relaxed); ///< Auto-assigned unique identifier.
  Node *parent_ = nullptr;                              ///< Parent in the component tree.
  uint32_t depth_ = 0;                                  ///< Depth in the component tree.
};

/// @brief A leaf component in the simulation compound graph.
///
/// @details Components are the active simulation entities. They own ports
/// and interact with the simulation engine. Event handlers are registered
/// directly on Ports via Port::set_handler().
class Component : public Node {
public:
  /// @brief Construct a component with the given name.
  /// @param[in] name Human-readable name for this component.
  explicit Component(std::string name) : Node(std::move(name)) {}
  ~Component() override = default;

  /// @brief Called once before simulation starts.
  virtual void initialize() {}

  /// @brief Called once after simulation ends.
  virtual void finalize() {}

  /// @brief Return the list of ports owned by this component (const).
  /// @returns Const reference to the port vector.
  const std::vector<std::unique_ptr<Port>> &ports() const { return ports_; }

  /// @brief Return the list of ports owned by this component (mutable).
  /// @returns Mutable reference to the port vector.
  std::vector<std::unique_ptr<Port>> &ports() { return ports_; }

  /// @brief Add a port to this component. Ownership is transferred.
  /// @param[in] port The port to add (ownership transferred).
  /// @returns Raw pointer to the added port.
  Port *add_port(std::unique_ptr<Port> port);

  /// @brief Find a port by its ID.
  /// @param[in] port_id The port ID to search for.
  /// @returns Pointer to the port, or nullptr if not found.
  Port *find_port(PortID port_id) const;

  /// @brief Return the partition this component is assigned to.
  /// @returns The partition ID, or INVALID_PARTITION_ID if unassigned.
  PartitionID partition_id() const { return partition_id_; }

  /// @brief Assign this component to a partition.
  /// @param[in] pid The partition ID to assign.
  void set_partition_id(PartitionID pid) { partition_id_ = pid; }

  /// @brief Return the partitioning weight.
  /// @returns The component's weight.
  uint32_t weight() const { return weight_; }

  /// @brief Set the partitioning weight.
  /// @param[in] w New weight value.
  void set_weight(uint32_t w) { weight_ = w; }

  /// @brief Return the simulation engine this component belongs to.
  /// @returns Pointer to the engine, or nullptr if not yet registered.
  SimulationEngine *engine() const { return engine_; }

  /// @brief Associate this component with a simulation engine.
  /// @param[in] e Pointer to the simulation engine.
  void set_engine(SimulationEngine *e) { engine_ = e; }

private:
  std::vector<std::unique_ptr<Port>> ports_;        ///< Owned ports.
  PartitionID partition_id_ = INVALID_PARTITION_ID; ///< Assigned partition.
  uint32_t weight_ = 1;                             ///< Partitioning weight.
  SimulationEngine *engine_ = nullptr;              ///< Owning engine.
};

/// @brief A component that contains child components (inclusion edges).
class CompositeComponent : public Component {
public:
  /// @brief Construct a composite component with the given name.
  /// @param[in] name Human-readable name.
  explicit CompositeComponent(std::string name) : Component(std::move(name)) {}
  ~CompositeComponent() override = default;

  /// @brief Add a child component. Sets the child's parent and depth.
  /// @param[in] child The child to add (ownership transferred).
  /// @returns Raw pointer to the added child.
  Component *add_child(std::unique_ptr<Component> child);

  /// @brief Return the list of child components.
  /// @returns Const reference to the children vector.
  const std::vector<std::unique_ptr<Component>> &children() const { return children_; }

  /// @brief Find a direct child by name.
  /// @param[in] name The name to search for.
  /// @returns Pointer to the child, or nullptr if not found.
  Component *find_child(const std::string &name) const;

  /// @brief Recursively collect all components in the subtree, including this composite.
  /// @param[out] out Vector to append components into.
  void collect_components(std::vector<Component *> &out);

  /// @brief Return the total number of descendants (recursive).
  /// @returns Count of all children, grandchildren, etc.
  uint32_t num_descendants() const;

private:
  std::vector<std::unique_ptr<Component>> children_; ///< Owned child components.
};

/// @brief A directional connection between two ports.
///
/// @details Links carry messages between ports and model communication latency.
/// They are the adjacency edges in the compound graph.
class Link {
public:
  /// @brief Construct a link between two ports.
  /// @param[in] id Unique link identifier.
  /// @param[in] src Source port.
  /// @param[in] dst Destination port.
  /// @param[in] latency Propagation delay in simulation ticks.
  Link(LinkID id, Port *src, Port *dst, Tick latency)
      : id_(id), src_(src), dst_(dst), latency_(latency) {}
  virtual ~Link() = default;

  /// @brief Return the unique link identifier.
  /// @returns The LinkID.
  LinkID id() const { return id_; }

  /// @brief Return the source port.
  /// @returns Pointer to the source port.
  Port *src() const { return src_; }

  /// @brief Return the destination port.
  /// @returns Pointer to the destination port.
  Port *dst() const { return dst_; }

  /// @brief Return the propagation latency in simulation ticks.
  /// @returns Latency value.
  Tick latency() const { return latency_; }

  /// @brief Send a message over this link. Routes to local queue or
  /// cross-partition inbox based on partition assignment.
  /// @param[in] msg The message to send (ownership transferred).
  virtual void send(std::unique_ptr<Message> msg);

  /// @brief Whether this link crosses a partition boundary.
  /// @retval true Source and destination are in different partitions.
  /// @retval false Both endpoints are in the same partition.
  bool is_cross_partition() const;

  /// @brief Return the link weight (for partitioning cut cost).
  /// @returns The weight value.
  uint32_t weight() const { return weight_; }

  /// @brief Set the link weight.
  /// @param[in] w New weight value.
  void set_weight(uint32_t w) { weight_ = w; }

private:
  LinkID id_;           ///< Unique link identifier.
  Port *src_;           ///< Source port endpoint.
  Port *dst_;           ///< Destination port endpoint.
  Tick latency_;        ///< Propagation delay in ticks.
  uint32_t weight_ = 1; ///< Partitioning cut weight.
};

/// @brief A named connection point on a Component.
///
/// @details Ports are the endpoints of Links. Each port belongs to a single component
/// and connects to at most one peer port via a link.
class Port : public Node {
public:
  /// @brief Construct a port.
  /// @param[in] name Human-readable port name.
  /// @param[in] port_id Unique port identifier within the component.
  /// @param[in] owner The component that owns this port.
  Port(std::string name, PortID port_id, Component *owner)
      : Node(std::move(name)), port_id_(port_id), owner_(owner) {
    set_parent(owner);
  }

  /// @brief Return the port identifier.
  /// @returns The PortID.
  PortID port_id() const { return port_id_; }

  /// @brief Return the component that owns this port.
  /// @returns Pointer to the owning component.
  Component *owner() const { return owner_; }

  /// @brief Return the link attached to this port.
  /// @returns Pointer to the link, or nullptr if unconnected.
  Link *link() const { return link_; }

  /// @brief Attach a link to this port.
  /// @param[in] lnk  Pointer to the link to attach.
  void set_link(Link *lnk) { link_ = lnk; }

  /// @brief The peer port at the other end of the link.
  /// @returns Pointer to the peer port, or nullptr if unconnected.
  Port *peer() const {
    if (link_ == nullptr)
      return nullptr;
    return (link_->src() == this) ? link_->dst() : link_->src();
  }

  /// @brief Send a message through this port's link.
  /// @param[in] msg The message to send (ownership transferred).
  void send(std::unique_ptr<Message> msg) {
    assert(link_ != nullptr);
    msg->header().src_port = port_id_;
    Port *p = peer();
    assert(p != nullptr);
    msg->header().dst_port = p->port_id();
    link_->send(std::move(msg));
  }

  /// @brief Return the port's reusable message-arrival event.
  /// @returns Pointer to the event.
  Event *event() { return &event_; }

  /// @brief Set the handler invoked when a message arrives at this port.
  /// @param h Handler callback.
  void set_handler(EventHandler h) { event_.set_handler(std::move(h)); }

private:
  PortID port_id_;                                  ///< Port identifier within the component.
  Component *owner_;                                ///< Owning component.
  Link *link_ = nullptr;                            ///< Attached link (nullptr if unconnected).
  Event event_{owner_, EventType::MESSAGE_ARRIVAL}; ///< Reusable event for message arrivals.
};

/// @brief A link that buffers messages in a timestamp-ordered queue.
///
/// @details Unlike the base Link which immediately routes messages through the
/// simulation engine, QueuedLink stores them internally. The receiving
/// component explicitly pops or drains messages when ready.
class QueuedLink : public Link {
public:
  /// @brief Construct a queued link with bounded capacity.
  /// @param[in] id Unique link identifier.
  /// @param[in] src Source port.
  /// @param[in] dst Destination port.
  /// @param[in] latency Propagation delay in simulation ticks.
  /// @param[in] capacity Maximum number of buffered messages.
  QueuedLink(LinkID id, Port *src, Port *dst, Tick latency, size_t capacity)
      : Link(id, src, dst, latency), queue_(capacity) {}

  /// @brief Enqueue a message without asserting on a full queue.
  /// @param[in] msg The message to enqueue (ownership transferred).
  /// @retval true Message was enqueued successfully.
  /// @retval false Queue is full; message was not enqueued.
  bool try_send(std::unique_ptr<Message> msg) {
    msg->header().latency = latency();
    return queue_.push(std::move(msg));
  }

  /// @brief Enqueue a message, asserting the queue is not full.
  /// @param[in] msg The message to enqueue (ownership transferred).
  void send(std::unique_ptr<Message> msg) override {
    msg->header().latency = latency();
    bool ok = queue_.push(std::move(msg));
    assert(ok && "QueuedLink: send on full queue");
  }

  /// @brief Pop the next message from the queue (asserts non-empty).
  /// @returns The oldest message.
  std::unique_ptr<Message> pop() { return queue_.pop(); }

  /// @brief Peek at the next message without removing it.
  /// @returns Pointer to the oldest message, or nullptr if empty.
  const Message *peek() const { return queue_.peek(); }

  /// @brief Check whether the queue is empty.
  /// @retval true No messages are buffered.
  /// @retval false At least one message is buffered.
  bool empty() const { return queue_.empty(); }

  /// @brief Check whether the queue is at capacity.
  /// @retval true Queue has reached its maximum capacity.
  /// @retval false Queue has room for more messages.
  bool full() const { return queue_.full(); }

  /// @brief Return the number of buffered messages.
  /// @returns Current queue size.
  size_t size() const { return queue_.size(); }

  /// @brief Return the maximum number of messages the queue can hold.
  /// @returns Queue capacity.
  size_t capacity() const { return queue_.capacity(); }

  /// @brief Return the arrival tick of the next message.
  /// @returns Tick of the oldest message, or TICK_MAX if empty.
  Tick next_message_time() const { return queue_.next_message_time(); }

  /// @brief Pop all messages with arrival_tick <= current_time.
  /// @param[in] current_time The current simulation tick.
  /// @param[out] out Vector to append ready messages into.
  void drain_ready(Tick current_time, std::vector<std::unique_ptr<Message>> &out) {
    while (!queue_.empty() && queue_.next_message_time() <= current_time) {
      out.push_back(queue_.pop());
    }
  }

private:
  MessageQueue queue_; ///< Bounded timestamp-ordered message buffer.
};

} // namespace simdojo

#endif // SIMDOJO_COMPONENT_H_
