// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.

#ifndef SIMDOJO_CLOCKED_H_
#define SIMDOJO_CLOCKED_H_

#include "simdojo/clock_domain.h"
#include "simdojo/component.h"
#include "simdojo/event_queue.h"
#include "simdojo/simulation.h"

#include <string>
#include <utility>

namespace simdojo {

/// @brief CRTP mixin for components that operate on a clock.
///
/// @details Derives clock period and phase from a ClockDomain. Subclasses
/// override clock_edge() which is called on each rising edge. Owns a
/// reusable Event that is enqueued into the engine at each tick.
///
/// @tparam Base A Component-derived type (i.e., Component, CompositeComponent).
template <typename Base> class Clocked : public Base {
public:
  Clocked(std::string name, const ClockDomain &domain) : Base(std::move(name)), domain_(domain) {}

  void initialize() override {
    running_ = true;
    this->engine()->schedule_event(&clock_event_, domain_.first_edge());
  }

  /// @brief Resume clocking from the next clock edge at or after the given tick.
  /// No-op if already running.
  void resume_clock(Tick after) {
    if (running_)
      return;
    running_ = true;
    Tick first = domain_.first_edge();
    if (after < first) {
      this->engine()->schedule_event(&clock_event_, first);
      return;
    }
    Tick elapsed = (after - domain_.phase_offset()) % period();
    Tick next = (elapsed == 0) ? after : after + (period() - elapsed);
    this->engine()->schedule_event(&clock_event_, next);
  }

  /// @brief Return whether the clock is currently running.
  bool running() const { return running_; }

  /// @brief Return the clock domain this component belongs to.
  /// @returns Reference to the ClockDomain.
  const ClockDomain &clock_domain() const { return domain_; }

  /// @brief Clock period in simulation ticks.
  /// @returns Period value.
  Tick period() const { return domain_.period(); }

  /// @brief Clock frequency in Hz.
  /// @returns Frequency value.
  uint64_t frequency() const { return domain_.frequency(); }

  /// @brief Called on each rising clock edge. Return true to continue clocking.
  virtual bool clock_edge(Tick now) = 0;

private:
  const ClockDomain &domain_; ///< Clock source for period/phase.
  /// @brief Reusable clock edge event. Handler re-enqueues on the next edge
  /// if clock_edge() returns true, otherwise stops the clock.
  Event clock_event_{this, EventType::TIMER_CALLBACK, [this](Tick now, Message *) {
                       if (clock_edge(now)) {
                         this->engine()->schedule_event(&clock_event_, now + period());
                       } else {
                         running_ = false;
                       }
                     }};
  bool running_ = false; ///< True while the clock is active.
};

} // namespace simdojo

#endif // SIMDOJO_CLOCKED_H_
