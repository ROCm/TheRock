// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <numeric>

#include "ck_tile/core/utility/env.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/host/kernel_launch.hpp"

namespace ck_tile {

template <index_t BlockSize, typename KernelArgs, typename KernelImpl>
CK_TILE_HOST index_t get_max_occupancy_for_kernel()
{
    constexpr int dynamic_smem_size = 0;
    constexpr int min_blocks_per_cu = 1;

    const auto kernel_ptr = kentry<min_blocks_per_cu, KernelImpl, KernelArgs>;

    int max_occupancy = 0;
    hip_check_error(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_occupancy, kernel_ptr, BlockSize, dynamic_smem_size));

    return static_cast<index_t>(max_occupancy);
}

CK_TILE_HOST index_t get_best_occupancy_k_batch_value(index_t max_occupancy, index_t grid_size)
{
    static const index_t num_cus = get_num_cus();
    const index_t max_capacity   = max_occupancy * num_cus;

    index_t k_batch          = 1;
    const auto optimal_split = static_cast<index_t>(std::floor((1.0 * max_capacity) / grid_size));
    if(optimal_split > 1)
    {
        k_batch = optimal_split;
    }

    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
    {
        std::cout << "[SPLIT-K AUTODEDUCE] Max active thread blocks per CU for GEMM kernel:  "
                  << max_occupancy << std::endl;
        std::cout << "[SPLIT-K AUTODEDUCE] Output grid size:  " << grid_size << std::endl;
        std::cout << "[SPLIT-K AUTODEDUCE] Optimal split-k value " << k_batch << std::endl;
    }
    return k_batch;
}

template <index_t BlockSize, typename KernelArgs, typename KernelImpl>
struct ActiveWorkgroupsPerCU
{
    CK_TILE_HOST ActiveWorkgroupsPerCU()
    {
        max_occupancy_ = get_max_occupancy_for_kernel<BlockSize, KernelArgs, KernelImpl>();
    }
    index_t max_occupancy_{1};
};

template <index_t BlockSize, typename KernelImpl, typename TilePartitioner, typename KernelArgs>
CK_TILE_HOST index_t calculate_optimal_k_batch(const KernelArgs& kargs)
{
    static ActiveWorkgroupsPerCU<BlockSize, KernelArgs, KernelImpl> active_workgroups_per_cu;

    const auto grid_size = TilePartitioner::GridSize(kargs.GemmM, kargs.GemmN) * kargs.GemmBatch;
    auto optimal_k_batch =
        get_best_occupancy_k_batch_value(active_workgroups_per_cu.max_occupancy_, grid_size);

    const auto max_allowed_k_batch = kargs.GemmK;
    optimal_k_batch                = std::min(optimal_k_batch, max_allowed_k_batch);

    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
    {
        std::cout << "[SPLIT-K AUTODEDUCE] Final k_batch value: " << optimal_k_batch << std::endl;
    }

    return optimal_k_batch;
}

} // namespace ck_tile
