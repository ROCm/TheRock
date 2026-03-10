# Simulation Framework Design

A Parallel Discrete Event Simulation (PDES) framework for modeling hardware
systems as hierarchical compound graphs [1]. Uses a conservative
Chandy-Misra-Bryant protocol [2][3] with Lower Bound on Time Stamp (LBTS)
[5] synchronization for safe parallel execution.

## File Overview

| File | Purpose |
|------|---------|
| `sim_types.h` | Tick, ComponentID, LinkID, PortID, PartitionID type aliases |
| `event_queue.h` | Event, EventQueueEntry, EventQueue, CrossPartitionQueue |
| `message.h` | MessageHeader, Message, TimestampAdvanceMessage, MessageQueue |
| `component.h` | Node, Component, CompositeComponent, Link, Port, QueuedLink |
| `component.cpp` | Implementation for Component, CompositeComponent, Link |
| `clock_domain.h` | ClockDomain (frequency, period, phase offset) |
| `clocked.h` | Clocked\<Base\> Curiously Recurring Template Pattern (CRTP) mixin for clock-driven components |
| `topology.h` | Topology, Partition, AdjacencyGraph, Partitioner |
| `topology.cpp` | Topology wiring, graph construction, multilevel Fiduccia-Mattheyses (FM) partitioning |
| `simulation.h` | PartitionContext, SimulationEngine |
| `simulation.cpp` | Engine run loop, LBTS computation, event dispatch |

---

## Simulation Time

All timestamps, periods, latencies, and phase offsets are expressed in
`Tick` (`uint64_t`). The tick resolution is chosen when creating a clock
domain — for example, a tick resolution of 1 GHz gives 1 ns precision.

Frequency is a separate dimension measured in Hz (`uint64_t`), not ticks.
The two are mixed in expressions like `ticks_per_second / frequency_hz` to
produce a `Tick` period, which is dimensionally correct.

---

## Compound Graph

The simulation topology is a **compound graph** [4] — a graph that combines
two kinds of relationships:

- **Inclusion edges** (parent-child): form the component tree, modeling
  physical containment (e.g., a GPU contains compute units, which contain
  ALUs)
- **Adjacency edges** (links): model communication channels between
  components, each with a propagation latency

This mirrors how hardware is structured: a hierarchy of blocks connected by
buses, crossbars, and point-to-point links.

### Components

**Component** is a leaf simulation entity that processes events. It owns
ports and interacts with the simulation engine. Event handlers are
registered directly on ports via `Port::set_handler()`.

**CompositeComponent** extends Component with a child list. Critically, a
composite can also handle events itself — it is not just a passive
container. This lets you model a block like a CPU that both contains
sub-blocks (ALUs, caches) and directly handles some events (e.g.,
interrupt routing) without introducing artificial wrapper components.

`collect_components()` flattens the entire subtree including composites,
so the engine initializes and dispatches to all components uniformly.

### Ports and Links

**Ports** are named connection points on a component. Each port owns a
reusable `Event` for message arrivals. Sending is done via `Port::send()`;
receiving is handled by calling `Port::set_handler()` to register a
callback that the engine invokes when a message arrives.

**Links** are **unidirectional** connections between two ports with a
propagation latency. Bidirectional communication requires two links, one
per direction. This matches hardware reality (request and response paths
are separate) and keeps timing, flow control, and routing simple.

```
Component0           Component1
  P0 ----link A----> P1       (request path, 10 tick latency)
  P3 <---link B----- P2       (response path, 5 tick latency)
```

**QueuedLink** is a variant that buffers messages in a bounded priority
queue instead of routing them through the engine immediately. The
receiving component explicitly drains messages when ready — useful for
pull-based consumption patterns like a clocked component that processes
its input queue each cycle.

---

## Clock Domains

A `ClockDomain` defines a shared clock source with a frequency, period, and
phase offset. Derived domains can be created with a frequency divider,
modeling clock hierarchies (e.g., a 1 GHz core clock deriving a 500 MHz
memory clock).

The `Clocked<Base>` CRTP mixin adds clock-driven behavior to any component
type. Subclasses override `clock_edge()`, which is called on each rising
edge. The mixin handles self-scheduling automatically.

```cpp
class ALU : public Clocked<Component> {
public:
  ALU(const ClockDomain &clk) : Clocked("alu", clk) {}
  bool clock_edge(Tick now) override { /* process one cycle */ return true; }
};
```

`Clocked` is templated on the base class, so it works with both `Component`
and `CompositeComponent`:

```cpp
class CPU : public Clocked<CompositeComponent> { ... };
```

---

## Messages and Events

### Messages

`Message` is the base class for data sent over links. Subclass it to add
payload for your model. Every message carries a `MessageHeader` with
timestamps, port IDs, and a sequence number.

`TimestampAdvanceMessage` is used by the LBTS protocol [2][3] — it carries
no payload and declares that no real message with an earlier timestamp will
be sent on this link, allowing the receiver to safely advance its LBTS.
Known in the literature as a "null message" [Chandy-Misra 1979, Bryant
1977].

### Events

Events are the scheduling primitive. There are four types, ordered by
priority (highest priority first):

1. `BARRIER_SYNC` — synchronization barrier pseudo-event
2. `TIMESTAMP_ADVANCE` — LBTS timestamp advance
3. `TIMER_CALLBACK` — scheduled timer/tick callback (used by `Clocked`)
4. `MESSAGE_ARRIVAL` — a message arrived at a port

Events with the same timestamp are ordered by type priority, then by
sequence number for deterministic tie-breaking.

### Zero-Allocation Event Model

`Event` is a long-lived, reusable descriptor holding a target component,
event type, and handler callback. Per-firing state (timestamp, message
payload) is stored in `EventQueueEntry`, not in the Event itself. The same
Event object can appear in the event queue multiple times at different
timestamps — no allocation is needed per scheduling.

- **Ports** own a reusable `Event` of type `MESSAGE_ARRIVAL`. Register a
  handler via `Port::set_handler()`.
- **Clocked** owns a reusable `Event` of type `TIMER_CALLBACK` that
  re-enqueues itself on each clock edge.
- **Timestamp advances** use the engine's shared `timestamp_advance_event_`
  (a handler-less Event of type `TIMESTAMP_ADVANCE`). They contribute to
  LBTS computation via their presence in the heap without executing a handler.

### Event Queues

Each partition has a thread-local `EventQueue` (a min-heap by timestamp).
Cross-partition events are staged in a `CrossPartitionQueue` — one per
(source, destination) partition pair, giving exactly one producer and one
consumer per queue (SPSC). Since push and drain are never concurrent —
the LBTS barrier serializes them — the queue is a plain vector with no
atomics or locks. The last arriving worker drains all partitions' incoming
queues inside the barrier while every other worker is blocked.

---

## Topology and Partitioning

The `Topology` owns the entire simulation graph: the root composite, all
links, clock domains, and partition assignments. It serves as the single
entry point for building and wiring a model.

### Building a Model

```cpp
Topology topo;

// Create clock domains.
auto *core_clk = topo.add_clock_domain("core", 1'000'000'000, 1'000'000'000);

// Build the component tree.
auto root = std::make_unique<CompositeComponent>("soc");
auto *gpu = root->add_child(std::make_unique<CompositeComponent>("gpu"));
auto *alu = gpu->add_child(std::make_unique<ALU>(*core_clk));
auto *mem = gpu->add_child(std::make_unique<MemController>(*core_clk));
topo.set_root(std::move(root));

// Wire ports with unidirectional links.
topo.add_link(alu_req_port, mem_req_port, /*latency=*/10);
topo.add_link(mem_resp_port, alu_resp_port, /*latency=*/5);
```

### Graph Partitioning

For parallel execution, the topology is partitioned into sub-graphs, one per
thread. The built-in partitioner uses a multilevel FM algorithm [6][7]:

1. **Coarsening** — heavy-edge matching contracts the graph level by level
2. **Initial bisection** — greedy assignment + FM refinement on the coarsest
   graph
3. **Uncoarsening** — project back, FM refine at each level
4. **k-way** — recursive bisection to reach the target partition count

The partitioner minimizes edge cut (cross-partition communication) while
balancing partition weights (configurable imbalance tolerance). No external
library dependencies.

---

## Simulation Engine

The `SimulationEngine` drives the simulation using a conservative
Chandy-Misra-Bryant protocol [2][3].

### Execution Model

- **Single-threaded** (`num_threads == 1`): the main thread runs the worker
  loop directly. No barriers, no service thread. Simplest mode for debugging
  and small models.

- **Multi-threaded** (`num_threads > 1`): N worker threads (one per
  partition) + the main thread as a service coordinator. Workers process
  events in parallel up to the current LBTS, then synchronize.

### LBTS Epoch

Each epoch follows this sequence:

1. Workers process all events with `timestamp <= current_lbts`
2. Workers send timestamp advances on quiescent cross-partition links
3. Workers arrive at a barrier; the last worker drains all incoming
   cross-partition queues
4. Service thread computes `new_lbts = min(next_event_time, min_outgoing)`
   across all partitions
5. Service thread runs registered service callbacks (watchdog, stats
   collection, progress reporting) while workers are still blocked
6. Service thread releases workers; workers resume processing

The LBTS is the global safe time — no event with a timestamp at or below
the LBTS will ever be generated, so all such events can be processed without
risk of causality violations.

### Timestamp Advances

In a conservative protocol, a partition can stall if it doesn't know whether
a neighbor might send an event with a smaller timestamp. Timestamp advances
(called "null messages" in the literature [2][3]) break this deadlock: after
processing its events, each worker sends a timestamp advance on all outgoing
cross-partition links with timestamp `current_lbts + link_latency`. This
advances the receiver's lower bound, guaranteeing progress.

### Service Callbacks

The engine supports registering callbacks that run on the service thread
once per LBTS epoch. Use these for periodic tasks that should not interfere
with event processing: watchdog timers, statistics snapshots, progress
reporting, or dynamic load monitoring.

---

## References

[1] R. M. Fujimoto, *Parallel and Distributed Simulation Systems*,
    Wiley-Interscience, 2000.

[2] K. M. Chandy and J. Misra, "Distributed Simulation: A Case Study in
    Design and Verification of Distributed Programs," *IEEE Transactions on
    Software Engineering*, vol. SE-5, no. 5, pp. 440-452, 1979.

[3] R. E. Bryant, "Simulation of Packet Communication Architecture Computer
    Systems," MIT-LCS-TR-188, Massachusetts Institute of Technology, 1977.

[4] G. Sugiyama and K. Misue, "Visualization of Structural Information:
    Automatic Drawing of Compound Digraphs," *IEEE Transactions on Systems,
    Man, and Cybernetics*, vol. 21, no. 4, pp. 876-892, 1991.

[5] F. Mattern, "Efficient Algorithms for Distributed Snapshots and Global
    Virtual Time Approximation," *Journal of Parallel and Distributed
    Computing*, vol. 18, no. 4, pp. 423-434, 1993.

[6] C. M. Fiduccia and R. M. Mattheyses, "A Linear-Time Heuristic for
    Improving Network Partitions," in *Proceedings of the 19th Design
    Automation Conference*, pp. 175-181, 1982.

[7] G. Karypis and V. Kumar, "A Fast and High Quality Multilevel Scheme for
    Partitioning Irregular Graphs," *SIAM Journal on Scientific Computing*,
    vol. 20, no. 1, pp. 359-392, 1998.
