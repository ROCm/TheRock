// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.

#include "simdojo/component.h"

#include "simdojo/simulation.h"

namespace simdojo {

Port *Component::add_port(std::unique_ptr<Port> port) {
  Port *raw = port.get();
  ports_.push_back(std::move(port));
  return raw;
}

Port *Component::find_port(PortID port_id) const {
  for (auto &p : ports_) {
    if (p->port_id() == port_id)
      return p.get();
  }
  return nullptr;
}

Component *CompositeComponent::add_child(std::unique_ptr<Component> child) {
  child->set_parent(this);
  child->set_depth(depth() + 1);
  Component *raw = child.get();
  children_.push_back(std::move(child));
  return raw;
}

Component *CompositeComponent::find_child(const std::string &name) const {
  for (auto &c : children_) {
    if (c->name() == name)
      return c.get();
  }
  return nullptr;
}

void CompositeComponent::collect_components(std::vector<Component *> &out) {
  out.push_back(this);
  for (auto &child : children_) {
    auto *composite = dynamic_cast<CompositeComponent *>(child.get());
    if (composite != nullptr) {
      composite->collect_components(out);
    } else {
      out.push_back(child.get());
    }
  }
}

uint32_t CompositeComponent::num_descendants() const {
  uint32_t count = 0;
  for (auto &child : children_) {
    count++;
    auto *composite = dynamic_cast<const CompositeComponent *>(child.get());
    if (composite != nullptr)
      count += composite->num_descendants();
  }
  return count;
}

bool Link::is_cross_partition() const {
  return src_->owner()->partition_id() != dst_->owner()->partition_id();
}

void Link::send(std::unique_ptr<Message> msg) {
  msg->header().latency = latency_;
  Tick arrival = msg->arrival_tick();

  Event *port_event = dst_->event();
  SimulationEngine *engine = src_->owner()->engine();
  assert(engine != nullptr);

  if (is_cross_partition()) {
    engine->send_cross_partition(src_->owner()->partition_id(), dst_->owner()->partition_id(),
                                 port_event, arrival, std::move(msg));
  } else {
    engine->schedule_event(port_event, arrival, std::move(msg));
  }
}

} // namespace simdojo
