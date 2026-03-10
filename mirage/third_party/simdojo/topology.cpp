// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.

#include "simdojo/topology.h"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>
#include <unordered_map>

namespace simdojo {

ClockDomain *Topology::add_clock_domain(std::string name, uint64_t frequency_hz,
                                        Tick ticks_per_second, Tick phase_offset) {
  auto domain =
      std::make_unique<ClockDomain>(std::move(name), frequency_hz, ticks_per_second, phase_offset);
  ClockDomain *raw = domain.get();
  clock_domains_.push_back(std::move(domain));
  return raw;
}

Link *Topology::add_link(Port *src, Port *dst, Tick latency, uint32_t weight) {
  auto link = std::make_unique<Link>(next_link_id_++, src, dst, latency);
  link->set_weight(weight);
  src->set_link(link.get());
  dst->set_link(link.get());
  Link *raw = link.get();
  links_.push_back(std::move(link));
  return raw;
}

std::vector<Component *> Topology::collect_all_components() const {
  std::vector<Component *> result;
  if (root_ != nullptr)
    root_->collect_components(result);
  return result;
}

uint32_t Topology::num_components() const {
  return static_cast<uint32_t>(collect_all_components().size());
}

void Topology::dfs_visit(std::function<void(Component *, uint32_t)> visitor) const {
  if (root_ == nullptr)
    return;

  struct Frame {
    Component *comp;
    uint32_t depth;
  };

  std::vector<Frame> stack;
  stack.push_back({root_.get(), 0});

  while (!stack.empty()) {
    auto [comp, depth] = stack.back();
    stack.pop_back();

    auto *composite = dynamic_cast<CompositeComponent *>(comp);
    if (composite != nullptr) {
      for (auto &child : composite->children()) {
        stack.push_back({child.get(), depth + 1});
      }
    } else {
      visitor(comp, depth);
    }
  }
}

void Topology::partition(uint32_t num_partitions) {

  if (num_partitions <= 1) {
    auto components = collect_all_components();
    Partition part;
    part.id = 0;
    part.components = std::move(components);
    for (auto *comp : part.components) {
      comp->set_partition_id(0);
      part.total_weight += comp->weight();
    }
    for (auto &link : links_)
      part.internal_links.push_back(link.get());
    partitions_.clear();
    partitions_.push_back(std::move(part));
    return;
  }

  auto adj = AdjacencyGraph::from_topology(*this);
  Partitioner::Config cfg;
  cfg.num_partitions = num_partitions;
  Partitioner partitioner(cfg);
  partitions_ = partitioner.partition(adj);

  // Classify links into internal or boundary for each partition.
  for (auto &link : links_) {
    PartitionID src_pid = link->src()->owner()->partition_id();
    PartitionID dst_pid = link->dst()->owner()->partition_id();
    if (src_pid == dst_pid) {
      partitions_[src_pid].internal_links.push_back(link.get());
    } else {
      partitions_[src_pid].boundary_links.push_back(link.get());
      partitions_[dst_pid].boundary_links.push_back(link.get());
      partitions_[src_pid].neighbor_partitions.insert(dst_pid);
      partitions_[dst_pid].neighbor_partitions.insert(src_pid);
    }
  }
}

AdjacencyGraph AdjacencyGraph::from_topology(const Topology &topo) {
  AdjacencyGraph g;
  auto components = topo.collect_all_components();
  g.num_nodes = static_cast<uint32_t>(components.size());
  g.adjacency.resize(g.num_nodes);
  g.node_weights.resize(g.num_nodes);
  g.index_to_component = components;

  std::unordered_map<ComponentID, uint32_t> id_to_index;
  for (uint32_t i = 0; i < g.num_nodes; ++i) {
    id_to_index[components[i]->id()] = i;
    g.node_weights[i] = components[i]->weight();
  }

  for (auto &link : topo.links()) {
    Component *src_comp = link->src()->owner();
    Component *dst_comp = link->dst()->owner();
    if (src_comp == dst_comp)
      continue;

    auto src_it = id_to_index.find(src_comp->id());
    auto dst_it = id_to_index.find(dst_comp->id());
    if (src_it == id_to_index.end() || dst_it == id_to_index.end())
      continue;

    uint32_t si = src_it->second;
    uint32_t di = dst_it->second;
    uint32_t w = link->weight();

    g.adjacency[si].emplace_back(di, w);
    g.adjacency[di].emplace_back(si, w);
  }

  return g;
}

std::vector<Partition> Partitioner::partition(AdjacencyGraph &graph) {
  std::vector<PartitionID> assignment(graph.num_nodes, 0);

  if (config_.num_partitions > 1)
    recursive_bisect(graph, assignment, 0, config_.num_partitions);

  // Assign partition IDs to components.
  for (uint32_t i = 0; i < graph.num_nodes; ++i)
    graph.index_to_component[i]->set_partition_id(assignment[i]);

  // Build Partition structs.
  std::vector<Partition> partitions(config_.num_partitions);
  for (uint32_t p = 0; p < config_.num_partitions; ++p)
    partitions[p].id = p;

  for (uint32_t i = 0; i < graph.num_nodes; ++i) {
    PartitionID pid = assignment[i];
    partitions[pid].components.push_back(graph.index_to_component[i]);
    partitions[pid].total_weight += graph.node_weights[i];
  }

  return partitions;
}

void Partitioner::recursive_bisect(AdjacencyGraph &graph, std::vector<PartitionID> &assignment,
                                   PartitionID base_id, uint32_t num_parts) {
  if (num_parts <= 1)
    return;

  // Multilevel FM bisection.
  std::vector<uint8_t> bisect(graph.num_nodes, 0);

  // Build coarsening hierarchy.
  std::vector<CoarsenLevel> levels;
  AdjacencyGraph current = graph;

  while (current.num_nodes > config_.coarsen_threshold && current.num_nodes > 2) {
    levels.push_back(coarsen(current));
    current = levels.back().graph;
    // Stop if coarsening made no progress (e.g., isolated nodes).
    if (current.num_nodes == levels.back().fine_to_coarse.size())
      break;
  }

  // Initial bisection on the coarsest graph.
  std::vector<uint8_t> coarse_assign(current.num_nodes, 0);
  fm_bisect(current, coarse_assign);

  // Uncoarsen and refine. At each level, project the assignment back to the
  // finer graph and refine on that finer graph's adjacency structure.
  for (int i = static_cast<int>(levels.size()) - 1; i >= 0; --i) {
    std::vector<uint8_t> fine_assign;
    uncoarsen(levels[i], coarse_assign, fine_assign);
    const AdjacencyGraph &fine_graph = (i > 0) ? levels[i - 1].graph : graph;
    fm_refine(fine_graph, fine_assign);
    coarse_assign = std::move(fine_assign);
  }

  // If we had coarsening levels, coarse_assign is for the original graph.
  // If no coarsening happened, coarse_assign is already for graph.
  bisect = std::move(coarse_assign);

  // Map bisection to partition IDs.
  uint32_t left_parts = num_parts / 2;
  uint32_t right_parts = num_parts - left_parts;

  // Collect sub-graph indices for recursive bisection.
  std::vector<uint32_t> left_indices;
  std::vector<uint32_t> right_indices;
  for (uint32_t i = 0; i < graph.num_nodes; ++i) {
    if (bisect[i] == 0) {
      assignment[i] = base_id;
      left_indices.push_back(i);
    } else {
      assignment[i] = base_id + left_parts;
      right_indices.push_back(i);
    }
  }

  // Recursively bisect each half if needed.
  auto build_subgraph = [&](const std::vector<uint32_t> &indices) -> AdjacencyGraph {
    AdjacencyGraph sub;
    sub.num_nodes = static_cast<uint32_t>(indices.size());
    sub.adjacency.resize(sub.num_nodes);
    sub.node_weights.resize(sub.num_nodes);
    sub.index_to_component.resize(sub.num_nodes);

    std::unordered_map<uint32_t, uint32_t> old_to_new;
    for (uint32_t ni = 0; ni < sub.num_nodes; ++ni) {
      uint32_t oi = indices[ni];
      old_to_new[oi] = ni;
      sub.node_weights[ni] = graph.node_weights[oi];
      sub.index_to_component[ni] = graph.index_to_component[oi];
    }

    for (uint32_t ni = 0; ni < sub.num_nodes; ++ni) {
      uint32_t oi = indices[ni];
      for (auto &[neighbor, w] : graph.adjacency[oi]) {
        auto it = old_to_new.find(neighbor);
        if (it != old_to_new.end())
          sub.adjacency[ni].emplace_back(it->second, w);
      }
    }
    return sub;
  };

  if (left_parts > 1) {
    auto left_graph = build_subgraph(left_indices);
    std::vector<PartitionID> left_assign(left_graph.num_nodes, 0);
    recursive_bisect(left_graph, left_assign, base_id, left_parts);
    for (uint32_t ni = 0; ni < left_graph.num_nodes; ++ni)
      assignment[left_indices[ni]] = left_assign[ni];
  }

  if (right_parts > 1) {
    auto right_graph = build_subgraph(right_indices);
    std::vector<PartitionID> right_assign(right_graph.num_nodes, 0);
    recursive_bisect(right_graph, right_assign, base_id + left_parts, right_parts);
    for (uint32_t ni = 0; ni < right_graph.num_nodes; ++ni)
      assignment[right_indices[ni]] = right_assign[ni];
  }
}

Partitioner::CoarsenLevel Partitioner::coarsen(const AdjacencyGraph &graph) {
  CoarsenLevel level;
  uint32_t n = graph.num_nodes;

  level.fine_to_coarse.resize(n, UINT32_MAX);

  // Random node ordering for matching.
  std::vector<uint32_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(42);
  std::shuffle(order.begin(), order.end(), rng);

  // Heavy-edge matching.
  std::vector<bool> matched(n, false);
  uint32_t coarse_id = 0;

  for (uint32_t idx : order) {
    if (matched[idx])
      continue;

    // Find the heaviest unmatched neighbor.
    uint32_t best_neighbor = UINT32_MAX;
    uint32_t best_weight = 0;
    for (auto &[neighbor, w] : graph.adjacency[idx]) {
      if (!matched[neighbor] && w > best_weight) {
        best_weight = w;
        best_neighbor = neighbor;
      }
    }

    if (best_neighbor != UINT32_MAX) {
      level.fine_to_coarse[idx] = coarse_id;
      level.fine_to_coarse[best_neighbor] = coarse_id;
      matched[idx] = true;
      matched[best_neighbor] = true;
    } else {
      level.fine_to_coarse[idx] = coarse_id;
      matched[idx] = true;
    }
    coarse_id++;
  }

  // Build coarse-to-fine mapping.
  level.coarse_to_fine.resize(coarse_id);
  for (uint32_t i = 0; i < n; ++i)
    level.coarse_to_fine[level.fine_to_coarse[i]].push_back(i);

  // Build coarse graph.
  AdjacencyGraph &cg = level.graph;
  cg.num_nodes = coarse_id;
  cg.adjacency.resize(coarse_id);
  cg.node_weights.resize(coarse_id, 0);
  cg.index_to_component.resize(coarse_id, nullptr);

  // Coarse node weights = sum of fine node weights.
  for (uint32_t ci = 0; ci < coarse_id; ++ci) {
    for (uint32_t fi : level.coarse_to_fine[ci])
      cg.node_weights[ci] += graph.node_weights[fi];
    // Representative component (first fine node).
    cg.index_to_component[ci] = graph.index_to_component[level.coarse_to_fine[ci][0]];
  }

  // Coarse edges: aggregate fine edges.
  for (uint32_t fi = 0; fi < n; ++fi) {
    uint32_t ci = level.fine_to_coarse[fi];
    for (auto &[neighbor, w] : graph.adjacency[fi]) {
      uint32_t cn = level.fine_to_coarse[neighbor];
      if (ci == cn)
        continue;

      // Check if edge already exists (linear scan, fine for small graphs).
      bool found = false;
      for (auto &[adj, aw] : cg.adjacency[ci]) {
        if (adj == cn) {
          aw += w;
          found = true;
          break;
        }
      }
      if (!found)
        cg.adjacency[ci].emplace_back(cn, w);
    }
  }

  return level;
}

void Partitioner::fm_bisect(const AdjacencyGraph &graph, std::vector<uint8_t> &assignment) {
  uint32_t n = graph.num_nodes;
  assignment.resize(n, 0);

  if (n <= 1)
    return;

  // Greedy initial assignment: assign to the lighter side, sorted by degree.
  // Sort nodes by degree (descending) for better initial balance.
  std::vector<uint32_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
    return graph.adjacency[a].size() > graph.adjacency[b].size();
  });

  uint64_t weight_0 = 0;
  uint64_t weight_1 = 0;
  for (uint32_t idx : order) {
    if (weight_0 <= weight_1) {
      assignment[idx] = 0;
      weight_0 += graph.node_weights[idx];
    } else {
      assignment[idx] = 1;
      weight_1 += graph.node_weights[idx];
    }
  }

  // FM refinement passes.
  for (uint32_t pass = 0; pass < config_.fm_max_passes; ++pass) {
    int64_t improvement = fm_refine(graph, assignment);
    if (improvement <= 0)
      break;
  }
}

int64_t Partitioner::fm_refine(const AdjacencyGraph &graph, std::vector<uint8_t> &assignment) {
  uint32_t n = graph.num_nodes;
  if (n <= 1)
    return 0;

  // Compute partition weights.
  uint64_t total_weight = 0;
  std::array<uint64_t, 2> part_weight = {0, 0};
  for (uint32_t i = 0; i < n; ++i) {
    part_weight[assignment[i]] += graph.node_weights[i];
    total_weight += graph.node_weights[i];
  }

  uint64_t target = total_weight / 2;
  uint64_t max_allowed =
      static_cast<uint64_t>((1.0 + config_.imbalance_tolerance) * static_cast<double>(target));

  // Compute gains: gain[i] = external_cost - internal_cost.
  // Moving node i to the other side reduces cut by gain[i].
  std::vector<int64_t> gain(n, 0);
  for (uint32_t i = 0; i < n; ++i) {
    int64_t external = 0;
    int64_t internal = 0;
    for (auto &[neighbor, w] : graph.adjacency[i]) {
      if (assignment[neighbor] != assignment[i])
        external += w;
      else
        internal += w;
    }
    gain[i] = external - internal;
  }

  // FM pass: try moving each node once.
  std::vector<bool> locked(n, false);
  std::vector<uint32_t> move_sequence;
  std::vector<int64_t> cumulative_gain;
  int64_t running_gain = 0;

  for (uint32_t step = 0; step < n; ++step) {
    // Find the unlocked node with highest gain that maintains balance.
    int64_t best_gain = std::numeric_limits<int64_t>::min();
    uint32_t best_node = UINT32_MAX;

    for (uint32_t i = 0; i < n; ++i) {
      if (locked[i])
        continue;

      uint8_t from = assignment[i];
      uint8_t to = 1 - from;

      // Check balance: would moving i violate the constraint?
      uint64_t new_to_weight = part_weight[to] + graph.node_weights[i];
      if (new_to_weight > max_allowed)
        continue;

      if (gain[i] > best_gain) {
        best_gain = gain[i];
        best_node = i;
      }
    }

    if (best_node == UINT32_MAX)
      break;

    // Move best_node.
    uint8_t from = assignment[best_node];
    uint8_t to = 1 - from;
    assignment[best_node] = to;
    locked[best_node] = true;
    part_weight[from] -= graph.node_weights[best_node];
    part_weight[to] += graph.node_weights[best_node];

    running_gain += best_gain;
    move_sequence.push_back(best_node);
    cumulative_gain.push_back(running_gain);

    // Update neighbors' gains.
    for (auto &[neighbor, w] : graph.adjacency[best_node]) {
      if (locked[neighbor])
        continue;

      // If neighbor is now on the same side as best_node's new side,
      // the edge became internal (gain decreases for neighbor).
      // If on opposite side, edge became external (gain increases).
      if (assignment[neighbor] == to) {
        gain[neighbor] -= 2 * static_cast<int64_t>(w);
      } else {
        gain[neighbor] += 2 * static_cast<int64_t>(w);
      }
    }
  }

  if (move_sequence.empty())
    return 0;

  // Find the prefix with the best cumulative gain.
  auto best_it = std::max_element(cumulative_gain.begin(), cumulative_gain.end());
  int64_t best_total_gain = *best_it;

  if (best_total_gain <= 0) {
    // Undo all moves.
    for (auto it = move_sequence.rbegin(); it != move_sequence.rend(); ++it)
      assignment[*it] = 1 - assignment[*it];
    return 0;
  }

  // Undo moves after the best prefix.
  size_t best_prefix = static_cast<size_t>(best_it - cumulative_gain.begin()) + 1;
  for (size_t i = move_sequence.size(); i > best_prefix; --i)
    assignment[move_sequence[i - 1]] = 1 - assignment[move_sequence[i - 1]];

  return best_total_gain;
}

void Partitioner::uncoarsen(const CoarsenLevel &level,
                            const std::vector<uint8_t> &coarse_assignment,
                            std::vector<uint8_t> &fine_assignment) {
  uint32_t fine_n = static_cast<uint32_t>(level.fine_to_coarse.size());
  fine_assignment.resize(fine_n);
  for (uint32_t i = 0; i < fine_n; ++i)
    fine_assignment[i] = coarse_assignment[level.fine_to_coarse[i]];
}

uint64_t Partitioner::compute_cut(const AdjacencyGraph &graph,
                                  const std::vector<uint8_t> &assignment) const {
  uint64_t cut = 0;
  for (uint32_t i = 0; i < graph.num_nodes; ++i) {
    for (auto &[neighbor, w] : graph.adjacency[i]) {
      if (assignment[i] != assignment[neighbor])
        cut += w;
    }
  }
  return cut / 2; // Each edge counted twice.
}

} // namespace simdojo
