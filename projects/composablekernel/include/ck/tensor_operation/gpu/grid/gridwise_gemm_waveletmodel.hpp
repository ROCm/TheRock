// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"

namespace ck {

template <typename TileLoadThreadGroup, index_t NumGemmKPrefetchStage>
struct GridwiseGemmLoadWave;

// 1-stage prefetch
template <typename TileLoadThreadGroup>
struct GridwiseGemmLoadWave<TileLoadThreadGroup, 1>
{
    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */)
    {
        // TODO: improve applicability
        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
    }

    template <bool HasMainLoop,
              typename AGridDesc,
              typename ABlockDesc,
              typename ABlockTransfer,
              typename AGridBuffer,
              typename ABlockBuffer,
              typename ABlockTransferStep,
              typename BGridDesc,
              typename BBlockDesc,
              typename BBlockTransfer,
              typename BGridBuffer,
              typename BBlockBuffer,
              typename BBlockTransferStep>
    static __device__ void RunLoadWavePipeline(const AGridDesc& a_grid_desc,
                                               const ABlockDesc& a_block_desc,
                                               ABlockTransfer& a_blockwise_copy,
                                               const AGridBuffer& a_grid_buf,
                                               ABlockBuffer& a_block_buf,
                                               const ABlockTransferStep& a_block_copy_step,
                                               const BGridDesc& b_grid_desc,
                                               const BBlockDesc& b_block_desc,
                                               BBlockTransfer& b_blockwise_copy,
                                               const BGridBuffer& b_grid_buf,
                                               BBlockBuffer& b_block_buf,
                                               const BBlockTransferStep& b_block_copy_step,
                                               index_t num_loop)
    {
        // global read 0
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        // move to 1
        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // LDS write 0
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                // sync for Load threads()
                block_sync_lds();
                // global read i + 1
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                // move to i + 2
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                // sync with math threads()
                block_sync_lds();

                // LDS write i+1
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();
            // GEMM num_loop - 1
        }
    }
};

template <typename TileMathThreadGroup, index_t NumGemmKPrefetchStage>
struct GridwiseGemmMathWave;
// 1- stage prefetch
template <typename TileMathThreadGroup>
struct GridwiseGemmMathWave<TileMathThreadGroup, 1>
{

    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */) { return true; }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
    }

    template <bool HasMainLoop,
              typename ABlockBuffer,
              typename BBlockBuffer,
              typename BlockwiseGemm,
              typename CThreadBuffer>
    static __device__ void RunMathWavePipeline(ABlockBuffer& a_block_buf,
                                               BBlockBuffer& b_block_buf,
                                               const BlockwiseGemm& block_gemm,
                                               CThreadBuffer& c_thread_buf,
                                               index_t num_loop)
    {
        // Initialize C
        c_thread_buf.Clear();

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                block_sync_lds();

                // GEMM i
                block_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();
                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            // GEMM num_loop - 1
            block_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
    }
};

} // namespace ck
