#ifndef MIRAGE_SIM_TOPOLOGY_CLUSTER_PROFILE_H_
#define MIRAGE_SIM_TOPOLOGY_CLUSTER_PROFILE_H_

#include <cstdint>
#include <string>
#include <vector>

#include "lib/sim/isa/common/wavefront_size.h"

namespace mirage::sim::topology {

struct PackageProfile {
  std::string package_id;
  std::string sku_name;
  std::string arch_name;
  std::string gfx_target;
  std::uint32_t gpu_count = 0;
  std::uint64_t total_hbm_bytes = 0;
  std::uint64_t total_vram_bytes = 0;
  std::string fabric_id;
  std::string fabric_kind;
};

struct FabricLinkProfile {
  std::string link_id;
  std::string fabric_id;
  std::string source_node_id;
  std::string source_gpu_id;
  std::string target_node_id;
  std::string target_gpu_id;
  std::string link_kind;
  std::uint32_t lane_count = 0;
  std::uint64_t bandwidth_bytes_per_second = 0;
  std::uint64_t latency_ns = 0;
};

struct GpuProfile {
  std::string gpu_id;
  std::string stable_device_id;
  std::string arch_name;
  std::string gfx_target;
  std::uint32_t compute_units = 0;
  std::uint32_t wavefront_size = isa::kWavefrontSize64;
  std::uint64_t hbm_bytes = 0;
  std::uint64_t vram_bytes = 0;

  std::uint64_t EffectiveVramBytes() const {
    return vram_bytes == 0 ? hbm_bytes : vram_bytes;
  }

  void NormalizeWavefrontSize() {
    wavefront_size = isa::DefaultWavefrontSizeForGfxTarget(gfx_target);
  }
};

struct NodeProfile {
  std::string node_id;
  std::string stable_node_id;
  std::vector<GpuProfile> gpus;

  std::uint32_t GpuCount() const {
    return static_cast<std::uint32_t>(gpus.size());
  }

  std::uint64_t TotalHbmBytes() const {
    std::uint64_t total_hbm_bytes = 0;
    for (const GpuProfile& gpu : gpus) {
      total_hbm_bytes += gpu.hbm_bytes;
    }
    return total_hbm_bytes;
  }

  std::uint64_t TotalVramBytes() const {
    std::uint64_t total_vram_bytes = 0;
    for (const GpuProfile& gpu : gpus) {
      total_vram_bytes += gpu.EffectiveVramBytes();
    }
    return total_vram_bytes;
  }
};

struct ClusterProfile {
  std::string cluster_id;
  PackageProfile package;
  std::vector<NodeProfile> nodes;
  std::vector<FabricLinkProfile> links;

  std::uint32_t MaterializedGpuCount() const {
    std::uint32_t gpu_count = 0;
    for (const NodeProfile& node : nodes) {
      gpu_count += node.GpuCount();
    }
    return gpu_count;
  }

  std::uint32_t GpuCount() const {
    return package.gpu_count == 0 ? MaterializedGpuCount() : package.gpu_count;
  }

  std::uint64_t TotalHbmBytes() const {
    if (package.total_hbm_bytes != 0) {
      return package.total_hbm_bytes;
    }

    std::uint64_t total_hbm_bytes = 0;
    for (const NodeProfile& node : nodes) {
      total_hbm_bytes += node.TotalHbmBytes();
    }
    return total_hbm_bytes;
  }

  std::uint64_t TotalVramBytes() const {
    if (package.total_vram_bytes != 0) {
      return package.total_vram_bytes;
    }

    std::uint64_t total_vram_bytes = 0;
    for (const NodeProfile& node : nodes) {
      total_vram_bytes += node.TotalVramBytes();
    }
    return total_vram_bytes;
  }
};

ClusterProfile CreateMi355xSinglePackageProfile();

}  // namespace mirage::sim::topology

#endif  // MIRAGE_SIM_TOPOLOGY_CLUSTER_PROFILE_H_
