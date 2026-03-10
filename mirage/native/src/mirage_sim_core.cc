#include "lib/sim/exec/dispatch/dispatch_context.h"
#include "lib/sim/fabric/fabric_model.h"
#include "lib/sim/gpu/gpu_properties.h"
#include "lib/sim/gpu/virtual_gpu_device.h"
#include "lib/sim/memory/gpu_va_space.h"
#include "lib/sim/memory/memory_region.h"
#include "lib/sim/queue/queue_state.h"
#include "lib/sim/replay/replay_log.h"
#include "lib/sim/timing/timing_model.h"
#include "lib/sim/topology/cluster_instance.h"
#include "lib/sim/topology/cluster_profile.h"
#include "lib/sim/trace/trace_sink.h"

#include <cstdint>
#include <string>

namespace mirage::sim::topology {
namespace {

constexpr std::uint32_t kMi355xGpuCount = 8;
constexpr std::uint32_t kMi355xComputeUnits = 256;
constexpr std::uint32_t kMi355xWavefrontSize = 64;
constexpr std::uint64_t kGigabyte = 1000ULL * 1000ULL * 1000ULL;
constexpr std::uint64_t kMi355xHbmBytesPerGpu = 288ULL * kGigabyte;
constexpr std::uint64_t kMi355xLinkBandwidthBytesPerSecond =
    153600000000ULL;
constexpr std::uint64_t kMi355xLinkLatencyNs = 250ULL;

}  // namespace

ClusterProfile CreateMi355xSinglePackageProfile() {
  ClusterProfile profile;
  profile.cluster_id = "cluster-mi355x";
  profile.package.package_id = "package0";
  profile.package.sku_name = "AMD Instinct MI355X Platform";
  profile.package.arch_name = "CDNA4";
  profile.package.gfx_target = "gfx950";
  profile.package.gpu_count = kMi355xGpuCount;
  profile.package.total_hbm_bytes =
      kMi355xGpuCount * kMi355xHbmBytesPerGpu;
  profile.package.total_vram_bytes = profile.package.total_hbm_bytes;
  profile.package.fabric_id = "package0/infinity-fabric0";
  profile.package.fabric_kind = "amd-infinity-fabric";

  NodeProfile node;
  node.node_id = "node0";
  node.stable_node_id = "cluster-mi355x/package0/nodes/node0";

  for (std::uint32_t gpu_index = 0; gpu_index < kMi355xGpuCount; ++gpu_index) {
    GpuProfile gpu;
    gpu.gpu_id = "gpu" + std::to_string(gpu_index);
    gpu.stable_device_id =
        node.stable_node_id + "/devices/" + gpu.gpu_id;
    gpu.arch_name = profile.package.arch_name;
    gpu.gfx_target = profile.package.gfx_target;
    gpu.compute_units = kMi355xComputeUnits;
    gpu.wavefront_size = kMi355xWavefrontSize;
    gpu.hbm_bytes = kMi355xHbmBytesPerGpu;
    gpu.vram_bytes = kMi355xHbmBytesPerGpu;
    node.gpus.push_back(std::move(gpu));
  }

  profile.nodes.push_back(std::move(node));

  for (std::uint32_t source_index = 0; source_index < kMi355xGpuCount;
       ++source_index) {
    for (std::uint32_t target_index = 0; target_index < kMi355xGpuCount;
         ++target_index) {
      if (source_index == target_index) {
        continue;
      }

      FabricLinkProfile link;
      link.fabric_id = profile.package.fabric_id;
      link.source_node_id = profile.nodes.front().node_id;
      link.source_gpu_id = "gpu" + std::to_string(source_index);
      link.target_node_id = profile.nodes.front().node_id;
      link.target_gpu_id = "gpu" + std::to_string(target_index);
      link.link_id = profile.package.fabric_id + "/links/" + link.source_gpu_id +
                     "-" + link.target_gpu_id;
      link.link_kind = "xgmi";
      link.lane_count = 1;
      link.bandwidth_bytes_per_second = kMi355xLinkBandwidthBytesPerSecond;
      link.latency_ns = kMi355xLinkLatencyNs;
      profile.links.push_back(std::move(link));
    }
  }

  return profile;
}

ClusterInstance CreateMi355xSinglePackageInstance() {
  return ClusterInstance::Materialize(CreateMi355xSinglePackageProfile());
}

}  // namespace mirage::sim::topology

namespace mirage::sim {

const char* MirageSimCoreVersion() {
  return "0.1.0";
}

}  // namespace mirage::sim
