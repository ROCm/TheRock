#ifndef MIRAGE_SIM_TOPOLOGY_CLUSTER_INSTANCE_H_
#define MIRAGE_SIM_TOPOLOGY_CLUSTER_INSTANCE_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "lib/sim/topology/cluster_profile.h"

namespace mirage::sim::topology {

struct PackageIdentity {
  std::string cluster_id;
  std::string package_id;
  std::string sku_name;
};

struct FabricIdentity {
  std::string cluster_id;
  std::string package_id;
  std::string fabric_id;
  std::string fabric_kind;
};

struct NodeIdentity {
  std::string cluster_id;
  std::string package_id;
  std::string node_id;
  std::string stable_node_id;
  std::uint32_t node_index = 0;
};

struct DeviceIdentity {
  std::string cluster_id;
  std::string package_id;
  std::string node_id;
  std::string stable_node_id;
  std::string gpu_id;
  std::string stable_device_id;
  std::uint32_t gpu_index = 0;
};

struct LinkIdentity {
  std::string cluster_id;
  std::string package_id;
  std::string fabric_id;
  std::string link_id;
};

struct PackageInstance {
  PackageIdentity identity;
  FabricIdentity fabric;
  std::string arch_name;
  std::string gfx_target;
  std::uint32_t gpu_count = 0;
  std::uint64_t total_hbm_bytes = 0;
  std::uint64_t total_vram_bytes = 0;
};

struct GpuInstance {
  DeviceIdentity identity;
  GpuProfile profile;
  std::uint64_t vram_bytes = 0;
};

struct NodeInstance {
  NodeIdentity identity;
  std::uint64_t total_hbm_bytes = 0;
  std::uint64_t total_vram_bytes = 0;
  std::vector<GpuInstance> gpus;
};

struct FabricLinkInstance {
  LinkIdentity identity;
  DeviceIdentity source;
  DeviceIdentity target;
  std::string link_kind;
  std::uint32_t lane_count = 0;
  std::uint64_t bandwidth_bytes_per_second = 0;
  std::uint64_t latency_ns = 0;
};

namespace detail {

inline std::string DefaultClusterId(std::string_view cluster_id) {
  return cluster_id.empty() ? "cluster0" : std::string(cluster_id);
}

inline std::string DefaultPackageId(std::string_view package_id) {
  return package_id.empty() ? "package0" : std::string(package_id);
}

inline std::string DefaultFabricId(std::string_view package_id,
                                   std::string_view fabric_id) {
  if (!fabric_id.empty()) {
    return std::string(fabric_id);
  }
  return std::string(package_id) + "/fabric0";
}

inline std::string DefaultNodeId(std::string_view node_id,
                                 std::uint32_t node_index) {
  if (!node_id.empty()) {
    return std::string(node_id);
  }
  return "node" + std::to_string(node_index);
}

inline std::string DefaultGpuId(std::string_view gpu_id, std::uint32_t gpu_index) {
  if (!gpu_id.empty()) {
    return std::string(gpu_id);
  }
  return "gpu" + std::to_string(gpu_index);
}

inline std::string DefaultStableNodeId(std::string_view stable_node_id,
                                       std::string_view cluster_id,
                                       std::string_view package_id,
                                       std::string_view node_id) {
  if (!stable_node_id.empty()) {
    return std::string(stable_node_id);
  }
  return std::string(cluster_id) + "/" + std::string(package_id) + "/nodes/" +
         std::string(node_id);
}

inline std::string DefaultStableDeviceId(std::string_view stable_device_id,
                                         std::string_view stable_node_id,
                                         std::string_view gpu_id) {
  if (!stable_device_id.empty()) {
    return std::string(stable_device_id);
  }
  return std::string(stable_node_id) + "/devices/" + std::string(gpu_id);
}

inline std::string DefaultLinkId(std::string_view link_id,
                                 std::string_view fabric_id,
                                 std::string_view source_gpu_id,
                                 std::string_view target_gpu_id) {
  if (!link_id.empty()) {
    return std::string(link_id);
  }
  return std::string(fabric_id) + "/links/" + std::string(source_gpu_id) + "-" +
         std::string(target_gpu_id);
}

inline std::uint64_t EffectiveVramBytes(const GpuProfile& gpu) {
  return gpu.EffectiveVramBytes();
}

}  // namespace detail

struct ClusterInstance {
  std::string cluster_id;
  PackageInstance package;
  std::vector<NodeInstance> nodes;
  std::vector<FabricLinkInstance> links;

  static ClusterInstance Materialize(const ClusterProfile& profile) {
    ClusterInstance instance;
    instance.cluster_id = detail::DefaultClusterId(profile.cluster_id);

    const std::string package_id =
        detail::DefaultPackageId(profile.package.package_id);
    const std::string fabric_id =
        detail::DefaultFabricId(package_id, profile.package.fabric_id);

    instance.package.identity.cluster_id = instance.cluster_id;
    instance.package.identity.package_id = package_id;
    instance.package.identity.sku_name = profile.package.sku_name;
    instance.package.fabric.cluster_id = instance.cluster_id;
    instance.package.fabric.package_id = package_id;
    instance.package.fabric.fabric_id = fabric_id;
    instance.package.fabric.fabric_kind = profile.package.fabric_kind;
    instance.package.arch_name = profile.package.arch_name;
    instance.package.gfx_target = profile.package.gfx_target;

    std::uint32_t materialized_gpu_count = 0;
    std::uint64_t materialized_hbm_bytes = 0;
    std::uint64_t materialized_vram_bytes = 0;

    for (std::uint32_t node_index = 0; node_index < profile.nodes.size();
         ++node_index) {
      const NodeProfile& node_profile = profile.nodes[node_index];

      NodeInstance node_instance;
      node_instance.identity.cluster_id = instance.cluster_id;
      node_instance.identity.package_id = package_id;
      node_instance.identity.node_index = node_index;
      node_instance.identity.node_id =
          detail::DefaultNodeId(node_profile.node_id, node_index);
      node_instance.identity.stable_node_id = detail::DefaultStableNodeId(
          node_profile.stable_node_id, instance.cluster_id, package_id,
          node_instance.identity.node_id);

      for (std::uint32_t gpu_index = 0; gpu_index < node_profile.gpus.size();
           ++gpu_index) {
        const GpuProfile& gpu_profile = node_profile.gpus[gpu_index];

        GpuInstance gpu_instance;
        gpu_instance.profile = gpu_profile;
        gpu_instance.identity.cluster_id = instance.cluster_id;
        gpu_instance.identity.package_id = package_id;
        gpu_instance.identity.node_id = node_instance.identity.node_id;
        gpu_instance.identity.stable_node_id =
            node_instance.identity.stable_node_id;
        gpu_instance.identity.gpu_index = gpu_index;
        gpu_instance.identity.gpu_id =
            detail::DefaultGpuId(gpu_profile.gpu_id, gpu_index);
        gpu_instance.identity.stable_device_id = detail::DefaultStableDeviceId(
            gpu_profile.stable_device_id, gpu_instance.identity.stable_node_id,
            gpu_instance.identity.gpu_id);
        gpu_instance.profile.gpu_id = gpu_instance.identity.gpu_id;
        gpu_instance.profile.stable_device_id =
            gpu_instance.identity.stable_device_id;
        if (gpu_instance.profile.arch_name.empty()) {
          gpu_instance.profile.arch_name = instance.package.arch_name;
        }
        if (gpu_instance.profile.gfx_target.empty()) {
          gpu_instance.profile.gfx_target = instance.package.gfx_target;
        }
        gpu_instance.profile.NormalizeWavefrontSize();
        gpu_instance.vram_bytes = detail::EffectiveVramBytes(gpu_instance.profile);

        node_instance.total_hbm_bytes += gpu_instance.profile.hbm_bytes;
        node_instance.total_vram_bytes += gpu_instance.vram_bytes;
        ++materialized_gpu_count;
        materialized_hbm_bytes += gpu_instance.profile.hbm_bytes;
        materialized_vram_bytes += gpu_instance.vram_bytes;
        node_instance.gpus.push_back(std::move(gpu_instance));
      }

      instance.nodes.push_back(std::move(node_instance));
    }

    instance.package.gpu_count = materialized_gpu_count;
    instance.package.total_hbm_bytes =
        profile.package.total_hbm_bytes == 0 ? materialized_hbm_bytes
                                             : profile.package.total_hbm_bytes;
    instance.package.total_vram_bytes =
        profile.package.total_vram_bytes == 0 ? materialized_vram_bytes
                                              : profile.package.total_vram_bytes;

    for (const FabricLinkProfile& link_profile : profile.links) {
      const GpuInstance* source =
          instance.FindDevice(link_profile.source_node_id, link_profile.source_gpu_id);
      const GpuInstance* target =
          instance.FindDevice(link_profile.target_node_id, link_profile.target_gpu_id);

      FabricLinkInstance link_instance;
      link_instance.identity.cluster_id = instance.cluster_id;
      link_instance.identity.package_id = package_id;
      link_instance.identity.fabric_id =
          link_profile.fabric_id.empty() ? fabric_id : link_profile.fabric_id;
      link_instance.source =
          source == nullptr ? DeviceIdentity{} : source->identity;
      link_instance.target =
          target == nullptr ? DeviceIdentity{} : target->identity;
      link_instance.identity.link_id = detail::DefaultLinkId(
          link_profile.link_id, link_instance.identity.fabric_id,
          link_instance.source.gpu_id.empty() ? link_profile.source_gpu_id
                                              : link_instance.source.gpu_id,
          link_instance.target.gpu_id.empty() ? link_profile.target_gpu_id
                                              : link_instance.target.gpu_id);
      link_instance.link_kind = link_profile.link_kind;
      link_instance.lane_count = link_profile.lane_count;
      link_instance.bandwidth_bytes_per_second =
          link_profile.bandwidth_bytes_per_second;
      link_instance.latency_ns = link_profile.latency_ns;
      instance.links.push_back(std::move(link_instance));
    }

    return instance;
  }

  std::uint32_t GpuCount() const { return package.gpu_count; }

  const NodeInstance* FindNode(std::string_view node_id) const {
    for (const NodeInstance& node : nodes) {
      if (node.identity.node_id == node_id) {
        return &node;
      }
    }
    return nullptr;
  }

  const NodeInstance* FindNodeByStableId(std::string_view stable_node_id) const {
    for (const NodeInstance& node : nodes) {
      if (node.identity.stable_node_id == stable_node_id) {
        return &node;
      }
    }
    return nullptr;
  }

  const GpuInstance* FindDevice(std::string_view node_id,
                                std::string_view gpu_id) const {
    const NodeInstance* node = FindNode(node_id);
    if (node == nullptr) {
      return nullptr;
    }

    for (const GpuInstance& gpu : node->gpus) {
      if (gpu.identity.gpu_id == gpu_id) {
        return &gpu;
      }
    }
    return nullptr;
  }

  const GpuInstance* FindDeviceByStableId(
      std::string_view stable_device_id) const {
    for (const NodeInstance& node : nodes) {
      for (const GpuInstance& gpu : node.gpus) {
        if (gpu.identity.stable_device_id == stable_device_id) {
          return &gpu;
        }
      }
    }
    return nullptr;
  }

  const FabricLinkInstance* FindLink(std::string_view link_id) const {
    for (const FabricLinkInstance& link : links) {
      if (link.identity.link_id == link_id) {
        return &link;
      }
    }
    return nullptr;
  }
};

inline ClusterInstance MaterializeCluster(const ClusterProfile& profile) {
  return ClusterInstance::Materialize(profile);
}

ClusterInstance CreateMi355xSinglePackageInstance();

}  // namespace mirage::sim::topology

#endif  // MIRAGE_SIM_TOPOLOGY_CLUSTER_INSTANCE_H_
