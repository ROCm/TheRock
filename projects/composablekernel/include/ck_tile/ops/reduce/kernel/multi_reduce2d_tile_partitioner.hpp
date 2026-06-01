// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

/// @brief TilePartitioner for 2D reduction operations
template <typename BlockShape_, bool ForceMultiBlock_ = false>
struct Reduce2dTilePartitioner
{
    using BlockShape = remove_cvref_t<BlockShape_>;

    static constexpr bool ForceMultiBlock = ForceMultiBlock_;

    static constexpr index_t MPerBlock = BlockShape::Block_M;
    static constexpr index_t NPerBlock = BlockShape::Block_N;

    CK_TILE_HOST_DEVICE Reduce2dTilePartitioner() noexcept = delete;

    /// @brief Construct partitioner with problem dimensions
    /// @param total_reduce_len Total number of element in the reduction dimension
    CK_TILE_HOST_DEVICE Reduce2dTilePartitioner(index_t total_reduce_len) noexcept
        : total_reduction_length(total_reduce_len)
    {
    }

    /// @brief Get output tile index for threadwise reduction
    /// @param block_idx Block index
    CK_TILE_DEVICE auto GetOutputTileIndex(index_t block_idx) const noexcept -> index_t
    {
        return amd_wave_read_first_lane(block_idx);
    }

    /// @brief Get output tile index and block local ID for multi-block reduction
    /// @param block_global_idx Global block index
    /// @param block_group_size Number of blocks per output tile
    /// @return Tuple of (tile_index, local_block_id)
    CK_TILE_DEVICE auto
    GetOutputTileIndexMultiBlock(index_t block_global_idx,
                                 index_t block_group_size) const noexcept -> tuple<index_t, index_t>
    {
        const index_t tile_idx  = amd_wave_read_first_lane(block_global_idx / block_group_size);
        const index_t local_idx = amd_wave_read_first_lane(block_global_idx % block_group_size);
        return make_tuple(tile_idx, local_idx);
    }

    /// @brief Calculate the number of iterations and the number of blocks required to perform the
    /// reduction
    /// @return Tuple of (number of iteration per thread, number of blocks used in the reduction)
    CK_TILE_HOST_DEVICE auto GetBlockGroupParams() const noexcept -> tuple<index_t, index_t>
    {
        index_t block_group_size = 1;
        index_t num_iters        = 0;

        if(!ForceMultiBlock)
        {
            // Single-block strategy: one block handles entire reduction
            block_group_size = 1;
            num_iters        = (total_reduction_length + NPerBlock - 1) / NPerBlock;
            return make_tuple(num_iters, block_group_size);
        }
        else
        {
            constexpr int max_block_group_size =
                128; // Maximum 128, as in CK. It balances between latency (i.e. limiting stalls
                     // when performing the atomic operation) and block parallelism.

            num_iters = (total_reduction_length + (NPerBlock * max_block_group_size) - 1) /
                        (NPerBlock * max_block_group_size);

            // This should only happen if reduce_total_length is 0 (empty tensor)
            if(num_iters == 0)
            {
#ifndef __HIP_DEVICE_COMPILE__
                // Warning only on host side
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    printf("Warning: reduce_total_length is 0, there is no data to process\n");
                }
#endif
                block_group_size = 1;
                return make_tuple(num_iters, block_group_size);
            }

            block_group_size =
                (total_reduction_length + (NPerBlock * num_iters) - 1) / (NPerBlock * num_iters);

            return make_tuple(num_iters, block_group_size);
        }
    }

    /// @brief Compute the input tile offset for the given thread, block index
    /// @param block_global_idx Global index of the block processing (part) of the reduction
    /// @param block_group_size Number of blocks taking part in the reduction
    /// @param num_iterations Total number of iteration per thread
    /// @return Tuple of (M offset, N offset) for the input tile
    CK_TILE_DEVICE auto
    GetInputTileOffsets(const index_t block_global_idx,
                        const index_t block_group_size,
                        const index_t num_iterations) const -> tuple<index_t, index_t>
    {
        const auto [tile_idx, local_idx] =
            GetOutputTileIndexMultiBlock(block_global_idx, block_group_size);

        const index_t m_offset = MPerBlock * tile_idx;
        const index_t n_offset = NPerBlock * num_iterations * local_idx;

        return make_tuple(m_offset, n_offset);
    }

    /// @brief Compute the output tile offset for the given operation and block group
    /// @param block_group_id Index of block group processing a batch of rows
    /// @return Output tile offset
    CK_TILE_DEVICE index_t GetOutputTileOffset(const index_t block_group_id) const
    {
        return MPerBlock * block_group_id;
    }

    private:
    index_t total_reduction_length;
};
} // namespace ck_tile
