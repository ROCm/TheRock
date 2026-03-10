// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.

#ifndef SIMDOJO_TOPOLOGY_H_
#define SIMDOJO_TOPOLOGY_H_

#include "simdojo/clock_domain.h"
#include "simdojo/component.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace simdojo {

class Topology;

/// @brief A partition of the simulation topology.
///
/// @details Each Partition holds the set of components assigned to a single
/// worker thread, along with the links internal to and crossing this partition.
struct Partition {
  PartitionID id = 0;                  ///< Partition identifier.
  std::vector<Component *> components; ///< Components assigned to this partition.
  std::vector<Link *> internal_links;  ///< Links where both endpoints are in this partition.
  std::vector<Link *> boundary_links;  ///< Links crossing into or out of this partition.
  uint64_t total_weight = 0;           ///< Sum of component weights.
  std::unordered_set<PartitionID> neighbor_partitions; ///< Partitions connected via boundary links.
};

/// @brief Adjacency representation for the partitioning algorithm.
///
/// @details Converts the compound graph into a flat undirected graph where:
///   - Nodes = all components (leaves and composites)
///   - Edges = links between ports of different components
///   - Edge weights = link weights (for cut cost)
///   - Node weights = component weights (for balance)
struct AdjacencyGraph {
  uint32_t num_nodes = 0; ///< Number of nodes (all components).
  std::vector<std::vector<std::pair<uint32_t, uint32_t>>>
      adjacency;                      ///< Per-node adjacency list: (neighbor_index, edge_weight).
  std::vector<uint32_t> node_weights; ///< Per-node weights for balance constraints.
  std::vector<Component *> index_to_component; ///< Maps graph index back to the Component.

  /// @brief Build an adjacency graph from a topology.
  /// @param topo The topology to convert.
  /// @returns A flat undirected AdjacencyGraph.
  static AdjacencyGraph from_topology(const Topology &topo);
};

/// @brief The simulation compound graph.
///
/// Topology owns all components (via the root CompositeComponent), all links,
/// and all partitions. It provides component creation, link creation,
/// graph traversal, and partitioning into sub-graphs for parallel execution.
class Topology {
public:
  Topology() = default;

  /// @brief Set the root composite component (ownership transferred).
  /// @param root The root of the component tree.
  void set_root(std::unique_ptr<CompositeComponent> root) { root_ = std::move(root); }

  /// @brief Return the root composite component.
  /// @returns Pointer to the root, or nullptr if not set.
  CompositeComponent *root() const { return root_.get(); }

  /// @brief Create a link between two ports with a given latency.
  /// @param src Source port.
  /// @param dst Destination port.
  /// @param latency Propagation delay in simulation ticks.
  /// @param weight Partitioning cut weight for this link.
  /// @returns Raw pointer to the created link.
  Link *add_link(Port *src, Port *dst, Tick latency, uint32_t weight = 1);

  /// @brief Return the list of all links.
  /// @returns Const reference to the link vector.
  const std::vector<std::unique_ptr<Link>> &links() const { return links_; }

  /// @brief Collect all components (including composites) into a flat vector.
  /// @returns Vector of pointers to all components in the tree.
  std::vector<Component *> collect_all_components() const;

  /// @brief Return the total number of components in the topology.
  /// @returns Component count.
  uint32_t num_components() const;

  /// @brief Return the list of partitions (const).
  /// @returns Const reference to the partition vector.
  const std::vector<Partition> &partitions() const { return partitions_; }

  /// @brief Return the list of partitions (mutable).
  /// @returns Mutable reference to the partition vector.
  std::vector<Partition> &partitions() { return partitions_; }

  /// @brief Perform DFS visiting each leaf component.
  /// @param visitor Callback invoked with (component, depth) for each leaf.
  void dfs_visit(std::function<void(Component *, uint32_t)> visitor) const;

  /// @brief Create and register a clock domain.
  /// @param[in] name              Human-readable domain name.
  /// @param[in] frequency_hz      Clock frequency in Hz.
  /// @param[in] ticks_per_second  Simulation tick resolution.
  /// @param[in] phase_offset      Phase offset in simulation ticks.
  /// @returns Pointer to the created ClockDomain.
  ClockDomain *add_clock_domain(std::string name, uint64_t frequency_hz, Tick ticks_per_second,
                                Tick phase_offset = 0);

  /// @brief Return the list of registered clock domains.
  /// @returns Const reference to the clock domain vector.
  const std::vector<std::unique_ptr<ClockDomain>> &clock_domains() const { return clock_domains_; }

  /// @brief Partition the topology for parallel execution.
  /// @param num_partitions Number of partitions to create (one per thread).
  void partition(uint32_t num_partitions);

private:
  std::unique_ptr<CompositeComponent> root_;                ///< Root of the component tree.
  std::vector<std::unique_ptr<Link>> links_;                ///< All links in the topology.
  std::vector<std::unique_ptr<ClockDomain>> clock_domains_; ///< Registered clock domains.
  std::vector<Partition> partitions_;                       ///< Computed partitions.
  LinkID next_link_id_ = 0; ///< Counter for auto-assigning link IDs.
};

/// @brief Graph partitioner using a multilevel Fiduccia-Mattheyses (FM)
/// approach.
///
/// Self-contained implementation (no external libraries). The algorithm:
/// 1. Coarsening: heavy-edge matching to contract the graph
/// 2. Initial partitioning: FM bisection on the coarsened graph
/// 3. Uncoarsening/refinement: project back, FM refine at each level
/// 4. k-way: recursive bisection
class Partitioner {
public:
  /// @brief Configuration parameters for graph partitioning.
  struct Config {
    uint32_t num_partitions = 1;       ///< Target number of partitions.
    double imbalance_tolerance = 0.05; ///< Max allowed imbalance ratio (0.05 = 5%).
    uint32_t fm_max_passes = 10;       ///< Maximum FM refinement passes per level.
    uint32_t coarsen_threshold = 100;  ///< Stop coarsening when graph is this small.
  };

  /// @brief Construct a partitioner with the given configuration.
  /// @param config Partitioning parameters.
  explicit Partitioner(Config config) : config_(config) {}

  /// @brief Partition the graph and assign PartitionIDs to components.
  /// @param graph The adjacency graph to partition (components are updated in-place).
  /// @returns Vector of Partition descriptors.
  std::vector<Partition> partition(AdjacencyGraph &graph);

private:
  struct CoarsenLevel {
    AdjacencyGraph graph;
    std::vector<uint32_t> fine_to_coarse;
    std::vector<std::vector<uint32_t>> coarse_to_fine;
  };

  CoarsenLevel coarsen(const AdjacencyGraph &graph);

  void fm_bisect(const AdjacencyGraph &graph, std::vector<uint8_t> &assignment);

  int64_t fm_refine(const AdjacencyGraph &graph, std::vector<uint8_t> &assignment);

  void uncoarsen(const CoarsenLevel &level, const std::vector<uint8_t> &coarse_assignment,
                 std::vector<uint8_t> &fine_assignment);

  void recursive_bisect(AdjacencyGraph &graph, std::vector<PartitionID> &assignment,
                        PartitionID base_id, uint32_t num_parts);

  uint64_t compute_cut(const AdjacencyGraph &graph, const std::vector<uint8_t> &assignment) const;

  Config config_; ///< Partitioning parameters.
};

} // namespace simdojo

#endif // SIMDOJO_TOPOLOGY_H_
